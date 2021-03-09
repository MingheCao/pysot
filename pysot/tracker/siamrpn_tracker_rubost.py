# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

import torch
from pysot import rubost_track

import cv2


class SiamRPNRBTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNRBTracker, self).__init__(model)
        self.longterm_state = False

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        z_crop = self.get_z_crop(img, bbox)

        self.model.template(z_crop)
        self.zf_gt = self.model.zf
        self.zf_global = self.model.zf

    def init_gm(self, img):

        instance_size = cfg.TRACK.INSTANCE_SIZE
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)

        # expend z_crop to x_crop size
        x_crop = self.get_subwindow_init(img, self.center_pos, instance_size,
                                         round(s_x), round(s_z), self.channel_average)
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])

        # cv2.imshow('.',x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8))
        # cv2.waitKey(0)
        X = rubost_track.scoremap_sample_reject(score, 5000)
        X[:, [1, 0]] = X[:, [0, 1]]
        self.gmm_gt = rubost_track.gmm_fit(X, 1)

        return {'instance_size': cfg.TRACK.INSTANCE_SIZE,
                'score_size': 25,
                'score': score,
                'x_crop': x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8),
                'best_score': 0
                }

    def get_subwindow_init(self, im, pos, model_sz, original_sz, s_z, avg_chans):

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        # patch
        im_patch[0:int((original_sz - s_z) / 2), :, :] = avg_chans
        im_patch[:, 0:int((original_sz - s_z) / 2), :] = avg_chans
        im_patch[int((original_sz + s_z) / 2):original_sz, :, :] = avg_chans
        im_patch[:, int((original_sz + s_z) / 2):original_sz, :] = avg_chans
        #

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def get_z_crop(self, img, bbox):
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z_crop

    def template(self, img, bbox):
        z_crop = self.get_z_crop(img, bbox)

        return self.model.template_rb(z_crop)

    @torch.no_grad()
    def template_upate(self, zf, weight):
        if weight > 1 or weight < 0:
            raise ValueError("weight must between 0 and 1.")
        for idx in range(len(zf)):
            self.zf_global[idx] = (1 - weight) * self.zf_global[idx] + weight * zf[idx]

        self.model.zf = self.zf_global

    def check_if_update(self, score, best_score):

        X = rubost_track.scoremap_sample_reject(score, 2000)
        X[:, [1, 0]] = X[:, [0, 1]]
        # print(len(X))
        if len(X) <= 15:
            update_state = False
            return update_state
        gmm = rubost_track.gmm_fit(X, 6)

        labels = rubost_track.ChineseWhispers_gm(gmm, 5)

        rubost_track.plot_results_cw(X, gmm.predict(X), gmm.means_, gmm.covariances_, labels, 'Gaussian Mixture')

        kldiv = rubost_track.KLdiv_gmm(gmm, self.gmm_gt)
        self.kldiv = kldiv

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            update_state = False
            return update_state
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            update_state = True

        if kldiv < 1.8:
            update_state = True
        else:
            update_state = False

        return update_state

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
                     cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)
        # best_idx = np.argmax(score)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        update_state = self.check_if_update(score, best_score)

        if update_state:
            self.template_upate(self.template(img, bbox), 0.05)
            pass
        else:
            self.template_upate(self.zf_gt, 0.1)
            pass

        return {
            'bbox': bbox,
            'best_score': best_score,
            'instance_size': instance_size,
            'score_size': score_size,
            'score': score,
            'pscore': pscore,
            'x_crop': x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8),
            'kldiv': self.kldiv,
            'update_state': update_state
        }
