# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

import json

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    datasets=glob.glob(args.dataset +'/*')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    for v_idx, dataset in enumerate(datasets):
        if not os.path.isdir(dataset):
            continue
        print(dataset)

        frames = sorted(glob.glob(dataset + '/*.jpg'))
        rects = {}
        first_frame = True
        for img in frames:
            frame = cv2.imread(img)
            if first_frame:
                init_rect = cv2.selectROI('aa', frame, False, False)
                tracker.init(frame, init_rect)
                first_frame = False
                rects['/'.join(img.split('/')[-2:])] = init_rect
            else:
                outputs = tracker.track(frame)
                bbox = list(map(int, outputs['bbox']))
                rects['/'.join(img.split('/')[-2:])] = bbox

                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)

                cv2.imshow('aa',frame)
                cv2.waitKey(10)

        with open(dataset + '/trackings.json', 'w') as json_file:
            json.dump(rects, json_file)
        print('saving json.')

    print('done.')

if __name__ == '__main__':
    main()
