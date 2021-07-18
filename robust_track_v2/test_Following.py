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

from robust_track_v2 import rb_utils
from robust_track_v2.rb_tracker_v2 import rb_tracker_v2
from robust_track_v2.siam_model import SiamModel

import json

def main(args):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = SiamModel()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker=rb_tracker_v2(model)

    video_name = args.video_name.split('/')[-1].split('.')[0]
    base_path = '/'.join(args.video_name.split('/')[:-1])
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(video_name, 200, 220)

    with open('/'.join(args.video_name.split('/')[:-1]) + '/Following.json') as f:
        json_info = json.load(f)
    first_frame = True

    respath=os.path.join(args.save_path,args.video_name.split('/')[-1]+ '.txt')
    try:
        rects = np.loadtxt(respath,delimiter=',')
        rects[0,:] = json_info[video_name]['init_rect']
    except:
        rects = np.zeros((len(json_info[video_name]['img_names']),4))
        rects[0, :] = json_info[video_name]['init_rect']

    start_frame=1
    pluse_frame=130000

    for idx, img in enumerate(sorted(json_info[video_name]['img_names'])):
        frame_num=img.split('/')[-1].split('.')[0]
        frame_num = idx + 1
        frame = cv2.imread(os.path.join(base_path,img))
        if int(frame_num) < start_frame:
            continue

        if first_frame:
            try:
                # gt_rects = np.loadtxt(args.video_name.replace('img', 'groundtruth_rect.txt'), delimiter=',',
                #                       dtype='int')
                # init_rect = gt_rects[int(frame_num), :]
                init_rect = json_info[video_name]['gt_rect'][start_frame - 1]
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False

        else:
            outputs = tracker.track(frame)
            tracker.frame_num= int(frame_num)
            rects[int(frame_num) - 1, :] = np.array(outputs['bbox'])

            # rb_utils.visualize_response3d(outputs, fig, '1,2,1', frame_num)

            bbox = list(map(int, outputs['bbox']))

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            frame= rb_utils.plot_search_area(outputs, frame)

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
    parser.add_argument('--video_name', default='/home/rislab/Workspace/pysot/testing_dataset/UAV123/person28', type=str,
                        help='videos or image files')
    parser.add_argument('--save_path', default='', type=str,
                        help='')
    args = parser.parse_args()

    args.save_path = '/home/rislab/Workspace/pysot/robust_track_v2/result/Following'
    # weightpath='/home/rislab/Workspace/pysot/experiments/siamrpn_alex_dwxcorr'
    weightpath='/home/rislab/Workspace/pysot/experiments/siamrpn_r50_l234_dwxcorr'

    args.config = weightpath+'/config.yaml'
    args.snapshot = weightpath+'/model.pth'

    path = '/home/rislab/Workspace/pysot/testing_dataset/Following'
    with open(path + '/Following.json') as f:
        json_info = json.load(f)

    UAV123=['group1_4', \
               'group2_1', 'group2_2', 'group2_3', 'group3_1', 'group3_2', \
               'group3_3', 'group3_4', 'person4_1', 'person4_2', 'person9', \
               'person19_1', 'person19_2', 'person11', 'person18', 'person20']

    OTB100 = ['Human3']

    UAV123 = ['group3_4']

    # dataset = 'person9'
    # args.video_name = os.path.join(path, dataset)
    # main(args)

    idx = 0
    # for dataset in sorted(json_info.keys()):
    for dataset in sorted(UAV123):
        args.video_name = os.path.join(path, dataset)
        idx += 1
        print(idx)
        print(dataset)
        main(args)

