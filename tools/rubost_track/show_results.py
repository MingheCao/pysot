import numpy as np
from glob import glob
import cv2
import os
import argparse
import json

COLOR = ((1, 0, 0),
         (0, 1, 0),
         (1, 0, 1),
         (1, 1, 0),
         (0  , 162/255, 232/255),
         (0.5, 0.5, 0.5),
         (0, 0, 1),
         (0, 1, 1),
         (136/255, 0  , 21/255),
         (255/255, 127/255, 39/255),
         (0, 0, 0))
COLOR=list(COLOR)
for idx,c in enumerate(COLOR):
    COLOR[idx]= (c[2]*255,c[1]*255,c[0]*255)
COLOR = tuple(COLOR)

def main(dataset,video_name):
    path='/home/rislab/Workspace/pysot/testing_dataset/' + dataset

    with open(os.path.join(path,dataset+'.json')) as f:
        json_info = json.load(f)

    images = json_info[video_name]['img_names']
    gt_rects=json_info[video_name]['gt_rect']

    ourpnp_rects=np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Ours(Siamrpn++)/' + video_name +'.txt',delimiter=',')
    ourpn_rects=np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Ours(Siamrpn)/' + video_name +'.txt',delimiter=',')
    rpnp_rects = np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Siamrpn++/' + video_name +'.txt',delimiter=',')
    rpn_rects = np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/Siamrpn/' + video_name +'.txt',delimiter=',')
    da_rects = np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/DaSiamrpn/' + video_name +'.txt',delimiter=',')
    update_rects = np.loadtxt('/home/rislab/Workspace/pysot/tools/results/UGV/UpdateNet/' + video_name +'.txt',delimiter=',')



    for im,gtrect,ourpnp,ourpn,rpnp,rpn,da,update in zip(images,gt_rects,ourpnp_rects,ourpn_rects,rpnp_rects,rpn_rects,da_rects,update_rects):
        frame=cv2.imread(os.path.join(path,im))

        frame = cv2.putText(frame, '#' + str(int(im.split('/')[-1].replace('.jpg',''))), (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # cv2.rectangle(frame, (int(gtrect[0]), int(gtrect[1])),
        #               (int(gtrect[0] + gtrect[2]), int(gtrect[1] + gtrect[3])),
        #               (0, 255, 0), 3)
        cv2.rectangle(frame, (int(ourpnp[0]), int(ourpnp[1])),
                      (int(ourpnp[0] + ourpnp[2]), int(ourpnp[1] + ourpnp[3])),
                      COLOR[0], 3)
        cv2.rectangle(frame, (int(rpnp[0]), int(rpnp[1])),
                      (int(rpnp[0] + rpnp[2]), int(rpnp[1] + rpnp[3])),
                      COLOR[1], 3)
        cv2.rectangle(frame, (int(ourpn[0]), int(ourpn[1])),
                      (int(ourpn[0] + ourpn[2]), int(ourpn[1] + ourpn[3])),
                      COLOR[2], 3)
        cv2.rectangle(frame, (int(rpn[0]), int(rpn[1])),
                      (int(rpn[0] + rpn[2]), int(rpn[1] + rpn[3])),
                      COLOR[3], 3)
        cv2.rectangle(frame, (int(da[0]), int(da[1])),
                      (int(da[0] + da[2]), int(da[1] + da[3])),
                      COLOR[4], 3)
        cv2.rectangle(frame, (int(update[0]), int(update[1])),
                      (int(update[0] + update[2]), int(update[1] + update[3])),
                      COLOR[5], 3)

        cv2.imshow('',frame)
        kk = cv2.waitKey(0) & 0xFF
        if kk == ord('q'):
            return
        elif kk == ord('s'):
            cv2.imwrite('/home/rislab/Workspace/pysot/rb_result/qualitive/'  +im.replace('/','_'),frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--dataset', type=str, default='UGV',help='config file')
    parser.add_argument('--video_name', default='210121_3', type=str,
                        help='videos or image files')
    args = parser.parse_args()
    main(args.dataset,args.video_name)