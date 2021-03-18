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

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    images = glob(os.path.join(video_name, '*.jp*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame,img

def main():
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

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(video_name, 200, 220)

    fig = plt.figure(figsize=(6, 2.5))
    plt.get_current_fig_manager().window.wm_geometry("+1100+220")
    # mng=plt.get_current_fig_manager()
    # mng.window.SetPosition((500, 0))

    total_frames=len(glob(os.path.join(args.video_name, '*.jp*')))
    rects = np.zeros((total_frames, 4))

    start_frame=1
    pluse_frame=200

    for frame,img in get_frames(args.video_name):
        frame_num=img.split('/')[-1].split('.')[0]
        if int(frame_num) < start_frame:
            continue

        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                gt_rects = np.loadtxt(args.video_name.replace('img', 'groundtruth_rect.txt'), delimiter=',',
                                      dtype='int')
                init_rect = gt_rects[int(frame_num), :]
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

            for bbox in outputs['proposal_bbox']:
                bbox = list(map(int, bbox))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (255, 255, 0), 5)

            cv2.imshow(video_name, frame)
            if int(frame_num) < pluse_frame:
                cv2.waitKey(1)

            else:
                cv2.waitKey(0)
                data_name = args.video_name.split('/')[-2]
                path=args.video_name.replace('testing_dataset/OTB100/'+data_name+'/img', \
                                             'rb_result/' + data_name + '/' + str(start_frame) + '_' + \
                                             str(pluse_frame)+ '.txt')
                # np.savetxt(path, rects)



if __name__ == '__main__':
    main()
