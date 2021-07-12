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

import imageSimilarity_reid

from tools.rubost_track import rubost_track
import json

def crop_bbox(img, bbox):
    bbox = np.array(bbox)
    bbox[np.where(bbox < 0)] = 0

    return img[int(bbox[1]):int(bbox[1] + bbox[3]),
           int(bbox[0]):int(bbox[0] + bbox[2]), :]

def main(args):

    video_name = args.video_name.split('/')[-1].split('.')[0]

    with open('/'.join(args.video_name.split('/')[:-1]) + '/UGV.json') as f:
        json_info = json.load(f)
    total_frames=len(json_info[video_name]['img_names'])
    scores = []

    first_frame = True

    cfg={}
    cfg['sys_device_ids']=(0,)
    cfg['resize_h_w'] = (256, 128)
    cfg['last_conv_stride'] =1
    cfg['ckpt_file'] = ''
    cfg['model_weight_file']='/home/rislab/Workspace/person-reid-triplet-loss-baseline/weights/cuhk03_stride1/model_weight.pth'
    sim=imageSimilarity_reid(cfg)

    for img in json_info[video_name]['img_names']:
        frame_num=img.split('/')[-1].split('.')[0]
        frame = cv2.imread('/'.join(args.video_name.split('/')[:-1])+'/' +img)

        if first_frame:
            try:
                init_rect = json_info[video_name]['init_rect']

                scores.append(0)
            except:
                exit()
            tracker.init(frame, init_rect)
        else:
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
                cv2.waitKey(0)

            data_name = args.video_name.split('/')[-1]
            path='/home/rislab/Workspace/pysot/rb_result/Ours(Siamrpn)/' + data_name + '.txt'
            # np.savetxt(path, rects,delimiter=',')

if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    args = parser.parse_args()

    main(args)
