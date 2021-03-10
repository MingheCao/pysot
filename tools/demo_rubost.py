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
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from pysot import rubost_track

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame,img

def visualize(outputs,fig,frame_num):
    instance_size=outputs['instance_size']
    score_size=outputs['score_size']
    score = outputs['score']
    respond = score.reshape(5, score_size, score_size).max(0)
    X = np.arange(0, score_size, 1)
    Y = np.arange(0, score_size, 1)
    X, Y = np.meshgrid(X, Y)

    ax1 = plt.subplot(121, projection='3d')
    ax1.cla()
    surf = ax1.plot_surface(X, Y, respond, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_zlim(0.0, 1.0)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter('{x:.01f}')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ##

    plt.pause(0.1)

def add_text_info(outputs,frame_num):
    instance_size=outputs['instance_size']
    score_size=outputs['score_size']
    score = outputs['score']
    respond = score.reshape(5, score_size, score_size).max(0)

    # add heatmap
    maxval = respond.max()
    minval = respond.min()
    responsemap = (respond - minval) / (maxval - minval) * 255
    heatmap = cv2.resize(responsemap.astype(np.uint8), (instance_size, instance_size), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                  cv2.CV_8U)

    # add texts
    frame_show = cv2.addWeighted(outputs['x_crop'], 0.7, heatmap, 0.3, 0)
    strshow = 'frame: ' + str(int(frame_num))
    frame_show = cv2.putText(frame_show, strshow, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
    strshow = 'bestscore:' + str(outputs['best_score']).split('.')[0] + '.' + str(outputs['best_score']).split('.')[1][:3]
    frame_show = cv2.putText(frame_show, strshow, (110, 15), cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
    strshow= 'kldiv:' + str(outputs['kldiv']).split('.')[0] + '.' + str(outputs['kldiv']).split('.')[1][:3]
    frame_show = cv2.putText(frame_show, strshow, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
    strshow='update:'+ str(outputs['update_state'])
    frame_show = cv2.putText(frame_show, strshow, (110, 30), cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return frame_show



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

    cv2.namedWindow('heatmap', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('heatmap', 650, 220)

    fig = plt.figure(figsize=(6, 2.5))
    plt.get_current_fig_manager().window.wm_geometry("+1100+220")
    # mng=plt.get_current_fig_manager()
    # mng.window.SetPosition((500, 0))

    start_frame=1
    pluse_frame=20

    for frame,img in get_frames(args.video_name):
        frame_num=img.split('/')[-1].split('.')[0]
        if int(frame_num) < start_frame:
            continue

        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False

            outputs=tracker.init_gm(frame)

            visualize(outputs, fig, frame_num)
        else:
            outputs = tracker.track(frame)

            visualize(outputs,fig,frame_num)
            frame_show=add_text_info(outputs,frame_num)

            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.imshow('heatmap', frame_show)
            if int(frame_num) < pluse_frame:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
