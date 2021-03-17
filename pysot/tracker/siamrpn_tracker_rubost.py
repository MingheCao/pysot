# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

import torch
from tools.rubost_track import rubost_track

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
        self.instance_sizes=[cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.LOST_INSTANCE_SIZE]

        self.thre_kldiv= 10
        self.state_update = True

        self.visualize = True
        self.CONFIDENCE_LOW=0.985

    def init_gm(self, img):

        # instance_size = cfg.TRACK.INSTANCE_SIZE
        # w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # s_z = np.sqrt(w_z * h_z)
        # s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        #
        # # expend z_crop to x_crop size
        # x_crop = self.get_subwindow_init(img, self.center_pos, instance_size,
        #                                  round(s_x), round(s_z), self.channel_average)
        # outputs = self.model.track(x_crop)
        # score = self._convert_score(outputs['cls'])
        #
        # score_size=25
        # score_map=score.reshape(5, score_size, score_size).max(0)

        # cv2.imshow('.',x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8))
        # cv2.waitKey(0)
        # X = rubost_track.scoremap_sample_reject(score_map, 2000)
        # self.gmm_gt = rubost_track.gmm_fit(X, 1)

        return { }

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

    def template_raw(self, img, bbox):
        z_crop = self.get_z_crop(img, bbox)
        return self.model.template_rb_raw(z_crop)

    @torch.no_grad()
    def template_upate(self, zf, weight):
        if weight > 1 or weight < 0:
            raise ValueError("weight must between 0 and 1.")
        for idx in range(len(zf)):
            self.zf_global[idx] = (1 - weight) * self.zf_global[idx] + weight * zf[idx]

        self.model.zf = self.zf_global

    def sample_scoremap(self,score_map):
        num_sample = 2000
        X = rubost_track.scoremap_sample_reject(score_map, num_sample)

        state_sampling=False if len(X) <= int(num_sample / 100) else True
        return X,state_sampling

    def get_seg_gmm(self,gmm,labels):
        label=np.unique(labels)

        seg_gmm=[]
        for i, lb in enumerate(label):
            idx = np.where(labels == lb)
            mu = gmm.means_[idx]
            weight = gmm.weights_[idx]
            cov=gmm.covariances_[idx]

            seg_gmm.append((idx,weight,mu,cov))

        return seg_gmm

    def cal_gms_meancov(self,X,gmm,seg_gmm):

        Y_=gmm.predict(X)

        meancov=[]
        for idx,wt,mu,cov in seg_gmm:
            points=np.empty((0,2), float)
            for lb in idx[0]:
                points=np.vstack((points,X[Y_==lb,:]))

            mean=points.mean(axis=0)
            cov=np.cov(points.T)

            v, w = np.linalg.eigh(cov)
            std = np.sqrt(v[1])

            meancov.append((mean,cov,std))

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(points[:,0],points[:,1])
            # plt.plot(mean[0], mean[1], 'X', color='b')
            # plt.xlim(-12.5, 12.5)
            # plt.ylim(-12.5, 12.5)

        return meancov

    # def cal_center_gms(self,X,gmm,labels,score_sz):
    #     label=np.unique(labels)
    #     means=np.zeros((len(label),2))
    #
    #     for i, lb in enumerate(label):
    #         idx=np.where(labels==lb)
    #         mu=gmm.means_[idx]
    #         weight=gmm.weights_[idx]
    #         weight/=weight.sum()
    #         mu=mu*weight.reshape(-1,1)
    #         means[i,:]=mu.sum(axis=0)
    #
    #     dist=means*means
    #     dist=dist.reshape(-1,2).sum(axis=1)
    #     center_label=label[np.argmin(dist)]
    #
    #     return center_label

    def segment_groups(self,score_map):
        X,state_sampling=self.sample_scoremap(score_map)
        gmm = rubost_track.gmm_fit(X, 6)
        labels = rubost_track.ChineseWhispers_gm(gmm, threhold=5, n_iter=5)
        self.state_ngroups = len(np.unique(labels))
        if self.state_ngroups <= 0:
            raise ValueError("Groups must greater than 1.")

        seg_gmm=self.get_seg_gmm(gmm,labels)
        meancov=self.cal_gms_meancov(X,gmm,seg_gmm)


        if self.visualize:
            center_label=labels[0]
            rubost_track.plot_results_cw(X, gmm.predict(X), gmm.means_, gmm.covariances_, labels, center_label, '1,2,2', 'Gaussian Mixture',meancov)

        return {'state_sampling':state_sampling}

    # def kldiv_state(self, score,score_size):
    #
    #     center_label=self.get_center_gms(X,gmm,labels,score_size)
    #     center_mean,center_cov=self.cal_center_gmm_meancov(X,gmm,center_label,labels)
    #     kldiv = rubost_track.KLdiv_gmm_index(gmm, self.gmm_gt, np.where(labels == center_label))

    def calc_penalty(self,pred_bbox,scale_z):
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
        return penalty

    def penalize_score(self,score,penalty,score_size,update_state):

        pscore = penalty * score

        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)

        # window
        if update_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        return pscore


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
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        best_score=score[np.argmax(score)]

        index=np.argmax(score.reshape(5, -1),axis=0)
        score_nms=score.reshape(5, -1)[index,np.arange(score_size*score_size)]
        score_map=score_nms.reshape(score_size,score_size)
        pred_bbox_nms=pred_bbox.reshape(4, 5, -1)[:, index, np.arange(score_size * score_size)]

        # if self.state_update ==True:
        #     self.template_upate(self.zf_gt, 0.01)

        if best_score <= self.CONFIDENCE_LOW:
            self.state_update = False
            self.state_lost= True

            penalty = self.calc_penalty(pred_bbox, scale_z)
            pscore = self.penalize_score(score, penalty, score_size, True)
            best_idx = np.argmax(pscore)
            best_score = score[best_idx]
            bbox = pred_bbox[:, best_idx] / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]
            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr

        else:
            res=self.segment_groups(score_map)
            if self.state_ngroups:
                self.state_update = True

                penalty = self.calc_penalty(pred_bbox, scale_z)
                pscore=self.penalize_score(score, penalty, score_size,True)
                best_idx = np.argmax(pscore)
                best_score = score[best_idx]
                bbox = pred_bbox[:, best_idx] / scale_z
                lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
                cx = bbox[0] + self.center_pos[0]
                cy = bbox[1] + self.center_pos[1]
                width = self.size[0] * (1 - lr) + bbox[2] * lr
                height = self.size[1] * (1 - lr) + bbox[3] * lr

            else:
                pass
                # if self.state_kldiv:
                # else:
                #
                #         # scale_x=score_size/s_x
                #         # cx = self.center_mean[0]/scale_x+self.center_pos[0]
                #         # cy = self.center_mean[1]/scale_x+self.center_pos[1]
                #         center_mean=self.center_mean+ score_size/2
                #         center_mean=center_mean.astype(int)
                #
                #         penalty = self.calc_penalty(pred_bbox, scale_z)
                #         bbox=pred_bbox_nms.reshape(4,score_size,score_size)[:,center_mean[1],center_mean[0]]/ scale_z
                #         idx=center_mean[1]*score_size+center_mean[0]
                #         lr = penalty[idx] * score_nms[idx] * cfg.TRACK.LR
                #         cx = bbox[0] + +self.center_pos[0]
                #         cy = bbox[1] + self.center_pos[1]
                #         width = self.size[0] * (1 - lr) + bbox[2] * lr
                #         height = self.size[1] * (1 - lr) + bbox[3] * lr
                #
                #         # cx = self.center_pos[0]
                #         # cy = self.center_pos[1]


        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if self.state_update:
            self.template_upate(self.template(img, bbox), 0.06)


        return {
            'bbox': bbox,
            'best_score': best_score,
            'instance_size': instance_size,
            'x_crop': x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8),
            'kldiv': 0.,
            'update_state': self.state_update,
            's_x':s_x,
            'score_map':score_nms.reshape(score_size,score_size)
        }
