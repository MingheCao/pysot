# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

import torch

from robust_track_v2 import rb_utils as utils

import cv2


class rb_tracker_v2(SiamRPNTracker):
    def __init__(self, model):
        super(rb_tracker_v2, self).__init__(model)
        self.longterm_state = False

        self.frame_num = 0

        self.visualize = True
        self.visualize_gmm = False
        self.save_img = False
        # self.CONFIDENCE_LOW = 0.985
        # self.CONFIDENCE_LOW = 0.5
        self.CONFIDENCE_LOW = 0.1


        self.instance_sizes = {x: cfg.TRACK.INSTANCE_SIZE + 60 * x for x in range(10)}

        self.state_reliable_cnt = 0

        self.center_std_thre = 2.2
        self.reliable_cnt_thre = 5

        self.neg_flt_rate = 0.1

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        z_crop = self.get_z_crop(img, bbox)

        self.model.template(z_crop)
        self.zf_gt = self.model.zf

        self.zf_trusts = []

        self.zf_distractors = []

        self.bbox = np.array([0,0,0,0])

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

    def crop_zf(self, img, bbox, margin='wide'):
        center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])
        ssize = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        if margin == 'wide':
            w_z = ssize[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(ssize)
            h_z = ssize[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(ssize)
        elif margin == 'narrow':
            w_z = ssize[0]
            h_z = ssize[1]

        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, channel_average)
        return z_crop

    def template(self, img, bbox):
        z_crop = self.get_z_crop(img, bbox)
        return self.model.template_rb(z_crop)

    # @torch.no_grad()
    # def template_upate(self, zf, weight):
    #     if weight > 1 or weight < 0:
    #         raise ValueError("weight must between 0 and 1.")
    #     for idx in range(len(zf)):
    #         self.zf_global[idx] = (1 - weight) * self.zf_global[idx] + weight * zf[idx]
    #
    #     self.model.zf = self.zf_global

    def segment_groups(self, score_map):
        X, state_sampling = utils.sample_scoremap(score_map)

        if not state_sampling:
            return 0, float('-1'), state_sampling

        gmm = utils.gmm_fit(X, 6)
        labels = utils.ChineseWhispers_gm(gmm, threhold=5, n_iter=5)
        self.state_ngroups = len(np.unique(labels))
        if self.state_ngroups <= 0:
            raise ValueError("Groups must greater than 1.")

        seg_gmm = utils.get_seg_gmm(gmm, labels)
        meancov, point_set, stds = utils.cal_gms_meancov(X, gmm, seg_gmm)
        repond_idx = utils.get_respond_idx(score_map, point_set)

        center_std = np.amax(np.array(stds))

        if self.visualize_gmm:
            center_label = labels[0]
            utils.plot_results_cw(X, gmm.predict(X), seg_gmm, meancov, '1,2,2', str(self.frame_num))

        return repond_idx, center_std, state_sampling

    def find_pbbox(self, score_nms, pred_bbox_nms, score_size, scale_z):
        penalty = self.calc_penalty(pred_bbox_nms, scale_z)
        pscore = self.penalize_score(score_nms, penalty, score_size)
        best_idx = np.argmax(pscore)
        best_score = score_nms[best_idx]
        bbox = pred_bbox_nms[:, best_idx] / scale_z
        lr = penalty[best_idx] * score_nms[best_idx] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        pbbox = [cx, cy, width, height]

        return penalty, pbbox

    def find_proposal_bbox(self, penalty, score_nms, pred_bbox_nms, repond_idx, score_size, scale_z):

        proposal_bbox = []
        if repond_idx != 0:  # sampling_state == True
            for cord in repond_idx:
                idx = np.ravel_multi_index(cord, (score_size, score_size))
                bb = pred_bbox_nms[:, idx] / scale_z

                lr = penalty[idx] * score_nms[idx] * cfg.TRACK.LR
                cx = bb[0] + self.center_pos[0]
                cy = bb[1] + self.center_pos[1]
                width = self.size[0] * (1 - lr) + bb[2] * lr
                height = self.size[1] * (1 - lr) + bb[3] * lr

                proposal_bbox.append([cx, cy, width, height])

        return proposal_bbox

    def merge_bbox(self, img, pbbox, proposal_bbox):

        def to_lurd(box):
            return [box[0] - box[2] / 2, box[1] - box[3] / 2,
                    box[0] + box[2] / 2, box[1] + box[3] / 2]

        filtered_bbox = []
        for bb in proposal_bbox:
            iou_score = utils.cal_iou(to_lurd(pbbox), to_lurd(bb))
            # print('iou: %f' %(iou_score))
            if iou_score <= 0.1:
                cx, cy, width, height = self._bbox_clip(bb[0], bb[1], bb[2],
                                                        bb[3], img.shape[:2])
                filtered_bbox.append([cx - width / 2, cy - height / 2, width, height])
        return filtered_bbox

    def score_bbox_nms(self, outputs, score_size, anchors):
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)
        best_score = score[np.argmax(score)]

        index = np.argmax(score.reshape(5, -1), axis=0)
        score_nms = score.reshape(5, -1)[index, np.arange(score_size * score_size)]
        pred_bbox_nms = pred_bbox.reshape(4, 5, -1)[:, index, np.arange(score_size * score_size)]

        return score_nms, pred_bbox_nms, best_score

    def calc_penalty(self, pred_bbox, scale_z):
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

    def penalize_score(self, score, penalty, score_size):
        pscore = penalty * score

        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = window.flatten()
        # window = np.tile(window.flatten(), self.anchor_num)

        # window
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 window * cfg.TRACK.WINDOW_INFLUENCE

        return pscore

    def xcorr_score_nms(self,feature,x_crop,score_size):
        outputs = self.model.corr(x_crop, feature)
        score = self._convert_score(outputs['cls'])
        index = np.argmax(score.reshape(5, -1), axis=0)
        return score.reshape(5, -1)[index, np.arange(score_size * score_size)]

    def penalize_score_neg(self, score, score_size,instance_size,method='rect'):
        if method == 'hanning':
            hanning = np.hanning(score_size)
            window = -np.outer(hanning, hanning) + 1
            window = window.flatten()
            # window
            pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     window * cfg.TRACK.WINDOW_INFLUENCE
        elif method == 'rect':
            width = self.bbox[2] / instance_size * score_size * self.neg_flt_rate
            height = self.bbox[3] / instance_size * score_size * self.neg_flt_rate

            xmin = int(score_size / 2 - width / 2)
            xmax = int(score_size / 2 + width / 2)
            ymin = int(score_size / 2 - height / 2)
            ymax = int(score_size / 2 + height / 2)

            pscore = score.reshape(score_size,score_size)
            pscore[ymin:ymax+1,xmin:xmax+1] = 0
            pscore =pscore.reshape(-1)
        return pscore

    def track_prepocess(self, img):
        instance_size = self.instance_sizes[0]
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

        outputs = self.model.corr(x_crop, self.zf_gt)
        score_nms_gt, pred_bbox_nms_gt, best_score = self.score_bbox_nms(outputs, score_size, anchors)
        score_map_gt = score_nms_gt.reshape(score_size, score_size)
        penalty, pbbox = self.find_pbbox(score_nms_gt, pred_bbox_nms_gt, score_size, scale_z)

        repond_idx, center_std, state_sampling = self.segment_groups(score_map_gt)
        proposal_bbox = self.find_proposal_bbox(penalty, score_nms_gt, pred_bbox_nms_gt, repond_idx, score_size, scale_z)
        filtered_ppbbox = self.merge_bbox(img, pbbox, proposal_bbox)

        state_reliable = False
        state_occlusion = False

        if center_std < self.center_std_thre:
            if not len(filtered_ppbbox):
                state_reliable = True

        # if best_score <= self.CONFIDENCE_LOW or state_sampling is False:
        if best_score <= self.CONFIDENCE_LOW:
            state_reliable = False
            state_occlusion = True

        if self.visualize:
            frame = utils.visualize_tracking_heated('heat1', utils.img_tensor2cpu(x_crop), score_map_gt, instance_size,
                                                    self.frame_num, best_score)
            utils.put_text_update_std('heat1', frame, center_std, state_reliable)

        # ---------------------------------------------
        if len(filtered_ppbbox):
            for idx, bb in enumerate(filtered_ppbbox):
                z_im = self.crop_zf(img, bb, margin='wide')
                self.zf_distractors.append(self.model.template_rb(z_im))
                if len(self.zf_distractors) > 3:
                    self.remove_similar_features(self.zf_distractors)

                if self.visualize:
                    cv2.imshow(str(idx), utils.img_tensor2cpu(z_im))
                    cv2.waitKey(1)

        score_nms_neg = []
        if not state_occlusion:
            for idx,ft in enumerate(self.zf_distractors):
                score_neg=self.xcorr_score_nms(ft,x_crop,score_size)
                # if not state_occlusion:
                #     score_neg = self.penalize_score_neg(score_neg,score_size,instance_size, method = 'rect')
                score_nms_neg.append(score_neg)
                if self.visualize:
                    frame = utils.visualize_tracking_heated('neg '+str(idx), utils.img_tensor2cpu(x_crop), score_neg.reshape(score_size, score_size),
                                                            instance_size,self.frame_num, 0, win_shift=[350*idx, 400])
                    if not state_occlusion:
                        width = self.bbox[2] / instance_size * score_size * self.neg_flt_rate
                        height = self.bbox[3] / instance_size * score_size * self.neg_flt_rate

                        xmin = int((score_size / 2 - width / 2) * instance_size / score_size)
                        xmax = int((score_size / 2 + width / 2) * instance_size / score_size)
                        ymin = int((score_size / 2 - height / 2) * instance_size / score_size)
                        ymax = int((score_size / 2 + height / 2) * instance_size / score_size)
                        cv2.rectangle(frame, (xmin, ymin),
                                      (xmax, ymax),
                                      (0, 0, 255), 2)
                    cv2.imshow('neg '+str(idx), frame)
                    cv2.waitKey(1)

        score_nms_pos=[]
        for idx,ft in enumerate(self.zf_trusts):
            score_pos = self.xcorr_score_nms(ft, x_crop, score_size)
            score_nms_pos.append(score_pos)
            if self.visualize:
                frame = utils.visualize_tracking_heated('heat_pos', utils.img_tensor2cpu(x_crop), score_pos.reshape(score_size, score_size),
                                                        instance_size,
                                                        self.frame_num, 0, win_shift=[350, 35])
                cv2.imshow('heat_pos', frame)
                cv2.waitKey(1)

        if not state_occlusion:
            score_nms_flt = score_nms_gt + np.sum(np.array(score_nms_pos), axis=0) \
                         - np.sum(np.array(score_nms_neg), axis=0) + \
                            max(0,(len(score_nms_neg) - len(score_nms_pos) -1))*score_nms_gt + score_nms_gt
            # score_nms_flt = score_nms_gt + np.sum(np.array(score_nms_pos), axis=0) \
            #              - np.sum(np.array(score_nms_neg), axis=0)
            score_nms_flt[np.where(score_nms_flt < 0)] = 0
            pscore2 = self.penalize_score(score_nms_flt, penalty, score_size)
        else:
            score_nms_flt = score_nms_gt + np.sum(np.array(score_nms_pos), axis=0)
            score_nms_flt[np.where(score_nms_flt < 0)] = 0
            pscore2 = score_nms_flt

        best_idx2 = np.argmax(pscore2)
        best_score2 = score_nms_flt[best_idx2]
        bbox = pred_bbox_nms_gt[:, best_idx2] / scale_z
        lr = penalty[best_idx2] * score_nms_flt[best_idx2] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        pbbox = [cx, cy, width, height]

        score_mapp2 = pscore2.reshape(score_size, score_size)
        frame = utils.visualize_tracking_heated('heat_final', utils.img_tensor2cpu(x_crop), score_mapp2, instance_size,
                                                self.frame_num, best_score2, win_shift=[700, 35])
        cv2.imshow('heat_final', frame)
        cv2.waitKey(1)
        # utils.visualize_response3d({'score_map':score_mapp2},[],'1,1,1',[])

        return {'pbbox':pbbox, 'proposal_bbox':filtered_ppbbox,
                's_x':s_x, 'score_map':score_map_gt, 'x_crop':x_crop,
                'state_reliable':state_reliable,'state_occlusion':state_occlusion,
                'score_nms_gt':score_nms_gt,'score_size':score_size}

    @torch.no_grad()
    def remove_similar_features(self, features):
        recent_ft = features[-1]
        dist = []
        for ft in features[0:-1]:
            if not isinstance(recent_ft, list):
                dist.append(torch.sum(torch.square(recent_ft - ft)))
            else:
                dis = 0
                for idx in range(len(recent_ft)):
                    dis += torch.sum(torch.square(recent_ft[idx] - ft[idx]))
                dist.append(dis)
        del features[np.argmin(dist)]
        return features

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        state = self.track_prepocess(img)

        if not state['state_reliable']:
            self.state_reliable_cnt = 0
        else:
            self.state_reliable_cnt += 1
            if self.state_reliable_cnt >= self.reliable_cnt_thre:
                z_im = self.crop_zf(img,self.bbox,margin='wide')
                zf_trust = self.model.template_rb(z_im)
                if len(self.zf_trusts) <= 1:
                    self.zf_trusts.append(zf_trust)
                else:
                    score_trust=self.xcorr_score_nms(zf_trust,state['x_crop'],state['score_size'])
                    cos_dist=utils.similarity_cosine(score_trust.reshape(1,-1),state['score_nms_gt'].reshape(1,-1))
                    if cos_dist>=0.9:
                        self.zf_trusts.pop(0)
                        self.zf_trusts.append(zf_trust)
                self.state_reliable_cnt = 0

                cv2.imshow('reliable_temp', utils.img_tensor2cpu(z_im))
                cv2.waitKey(1)

        if not state['state_occlusion']:
        # if True:
            pbbox = state['pbbox']
            self.center_pos = np.array([pbbox[0], pbbox[1]])
            self.size = np.array([pbbox[2], pbbox[3]])

            cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                pbbox[3], img.shape[:2])

            self.bbox = [cx - width / 2,
                             cy - height / 2,
                             width,
                             height]

            self.state_scaled = False
        else:
            if not self.state_scaled:
                scal = 1.3
                self.size *= scal

                self.state_scaled = True
            else:
                pass

        state['bbox'] = self.bbox

        cv2.waitKey(1)
        return state