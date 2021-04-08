from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import matplotlib.pyplot as plt

from tools.rubost_track import rubost_track

import json

def get_frames(video_name):
    images = glob(os.path.join(video_name, '*.jp*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame,img

def main(args):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    video_name = args.video_name.split('/')[-1].split('.')[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(video_name, 200, 220)

    respath=os.path.join(args.save_path,args.video_name.split('/')[-1]+ '.txt')
    rects = np.loadtxt(respath,delimiter=',')

    with open('/'.join(args.video_name.split('/')[:-1]) + '/UGV.json') as f:
        json_info = json.load(f)
    first_frame = True

    rects[0,:] = json_info[video_name]['init_rect']

    start_frame=250
    pluse_frame=100000

    for frame,img in get_frames(args.video_name):
        frame_num=img.split('/')[-1].split('.')[0]
        if int(frame_num) < start_frame:
            continue

        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                # gt_rects = np.loadtxt(args.video_name.replace('img', 'groundtruth_rect.txt'), delimiter=',',
                #                       dtype='int')
                # init_rect = gt_rects[int(frame_num), :]
                init_rect = json_info[video_name]['gt_rect'][start_frame - 1]
                rects[0,:] = init_rect
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False

        else:
            outputs = tracker.track(frame)
            tracker.frame_num= int(frame_num)
            rects[int(frame_num) - 1, :] = np.array(outputs['bbox'])

            # rubost_track.visualize_response3d(outputs, fig, '1,2,1', frame_num)

            bbox = list(map(int, outputs['bbox']))

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            frame= rubost_track.plot_search_area(outputs, frame)

            if 'proposal_bbox' in outputs:
                for bbox in outputs['proposal_bbox']:
                    bbox = list(map(int, bbox))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (255, 255, 0), 5)

            cv2.imshow(video_name, frame)
            if int(frame_num) < pluse_frame:
                cv2.waitKey(1)
            else:
                break

    np.savetxt(respath, rects,delimiter=',')

if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--video_name', default='/home/rislab/Workspace/pysot/testing_dataset/UGV/210121_2', type=str,
                        help='videos or image files')
    parser.add_argument('--save_path', default='/home/rislab/Workspace/pysot/rb_result/UGV_results', type=str,
                        help='')
    args = parser.parse_args()

    main(args)