import numpy as np
from glob import glob
import cv2
import os
import argparse
import json

def main(dataset,video_name):
    path='/home/rislab/Workspace/pysot/testing_dataset/' + dataset

    with open(os.path.join(path,dataset+'.json')) as f:
        json_info = json.load(f)

    images = json_info[video_name]['img_names']
    gt_rects=json_info[video_name]['gt_rect']

    our_rects=np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Ours(Resnet)/' + video_name +'.txt',delimiter=',')
    our2_rects=np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Ours(Alexnet)/' + video_name +'.txt',delimiter=',')
    rpn_rects = np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Siamrpn(Resnet)/' + video_name +'.txt',delimiter=',')

    for im,gtrect,ourrect,our2rect,rpn_rect in zip(images,gt_rects,our_rects,our2_rects,rpn_rects):
        frame=cv2.imread(os.path.join(path,im))

        frame = cv2.putText(frame, im.split('/')[-1].replace('.jpg',''), (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.rectangle(frame, (int(gtrect[0]), int(gtrect[1])),
                      (int(gtrect[0] + gtrect[2]), int(gtrect[1] + gtrect[3])),
                      (0, 255, 0), 3)
        cv2.rectangle(frame, (int(ourrect[0]), int(ourrect[1])),
                      (int(ourrect[0] + ourrect[2]), int(ourrect[1] + ourrect[3])),
                      (0, 0, 255), 3)
        cv2.rectangle(frame, (int(our2rect[0]), int(our2rect[1])),
                      (int(our2rect[0] + our2rect[2]), int(our2rect[1] + our2rect[3])),
                      (255, 0, 0), 3)
        cv2.rectangle(frame, (int(rpn_rect[0]), int(rpn_rect[1])),
                      (int(rpn_rect[0] + rpn_rect[2]), int(rpn_rect[1] + rpn_rect[3])),
                      (0, 255, 255), 3)

        cv2.imshow('',frame)
        cv2.waitKey(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--dataset', type=str, default='UGV',help='config file')
    parser.add_argument('--video_name', default='210121_2', type=str,
                        help='videos or image files')
    args = parser.parse_args()
    main(args.dataset,args.video_name)