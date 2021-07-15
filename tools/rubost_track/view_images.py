
import os
import argparse

import cv2
from glob import glob

def get_frames(video_name):
    images = glob(os.path.join(video_name, '*.jp*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame,img

def main(args):
    vname=args.path.split('/')[-1]
    for frame,img in get_frames(args.path):
        frame_num=img.split('/')[-1].split('.')[0]
        cv2.imshow(vname, frame)
        kk = cv2.waitKey(10) & 0xFF
        if kk == ord('q'):
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--path', default='', type=str,
                        help='videos or image files')
    args = parser.parse_args()
    main(args)
