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
from tools.rubost_track import image_similarity

import cv2


class SiamRPNRBTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNRBTracker, self).__init__(model)
        self.longterm_state = False

        self.frame_num = 0
        self.state_update = True

        self.visualize = True
        self.visualize_gmm = False
        self.CONFIDENCE_LOW = 0.985
        self.instance_sizes = [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.LOST_INSTANCE_SIZE]

        self.state_update = False
        self.state_cross = False
        self.state_lost = False

        self.center_std_thre=2.5
        self.cross_count = 0

        self.img_similar=image_similarity.ImageSimilarity()

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

        self.zf_bbox_global=self.img_similar.get_feature(
            self.crop_bbox(img,bbox))


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

    def sample_scoremap(self, score_map):
        num_sample = 2000
        X = rubost_track.scoremap_sample_reject(score_map, num_sample)

        state_sampling = False if len(X) <= int(num_sample / 100) else True
        return X, state_sampling

    def get_seg_gmm(self, gmm, labels):
        label = np.unique(labels)

        seg_gmm = []
        wt_sum = []
        for i, lb in enumerate(label):
            idx = np.where(labels == lb)
            mu = gmm.means_[idx]
            weight = gmm.weights_[idx]
            cov = gmm.covariances_[idx]

            seg_gmm.append((idx, weight, mu, cov))
            wt_sum.append(weight.sum())

        wt_sum = np.array(wt_sum)
        index = np.argsort(wt_sum)[::-1]  # sort decent order
        seg_gmm_sort = []
        for idx in index:
            seg_gmm_sort.append(seg_gmm[idx])

        return seg_gmm_sort

    def cal_gms_meancov(self, X, gmm, seg_gmm):

        Y_ = gmm.predict(X)

        meancov = []
        point_set = []
        for idx, wt, mu, cov in seg_gmm:
            points = np.empty((0, 2), float)
            for lb in idx[0]:
                points = np.vstack((points, X[Y_ == lb, :]))

            mean = points.mean(axis=0)
            cov = np.cov(points.T)

            v, w = np.linalg.eigh(cov)
            std = np.sqrt(v[1])

            meancov.append((mean, cov, std))
            point_set.append(points)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(points[:,0],points[:,1])
            # plt.plot(mean[0], mean[1], 'X', color='b')
            # plt.xlim(-12.5, 12.5)
            # plt.ylim(-12.5, 12.5)

        return meancov, point_set

    def get_respond_idx(self, score_map, point_set, seg_gmm):
        score_size = score_map.shape[0]
        max_rep_idx = []
        for i in range(len(point_set)):
            if i > 1:
                break
            wt_sum = seg_gmm[i][1].sum()
            if wt_sum <= 0.3:
                continue

            points = point_set[i] + score_size / 2
            points = np.round(points).astype(np.int32)
            z = score_map[points[:, 1], points[:, 0]]
            pt = points[z.argmax(), :]
            max_rep_idx.append(np.array([pt[1], pt[0]]))

        return max_rep_idx

    def segment_groups(self, score_map):
        X, self.state_sampling = self.sample_scoremap(score_map)

        if not self.state_sampling:
            return 0

        gmm = rubost_track.gmm_fit(X, 6)
        labels = rubost_track.ChineseWhispers_gm(gmm, threhold=5, n_iter=5)
        self.state_ngroups = len(np.unique(labels))
        if self.state_ngroups <= 0:
            raise ValueError("Groups must greater than 1.")

        seg_gmm = self.get_seg_gmm(gmm, labels)
        meancov, point_set = self.cal_gms_meancov(X, gmm, seg_gmm)
        repond_idx = self.get_respond_idx(score_map, point_set, seg_gmm)

        mindist=float('inf')
        for mean, _, std in meancov[:2]:
            dist=mean*mean
            dist=dist.sum()
            if dist<mindist:
                mindist=dist
                self.center_std=std

        self.state_std = True if self.center_std < self.center_std_thre else False

        if self.visualize_gmm:
            center_label = labels[0]
            rubost_track.plot_results_cw(X, gmm.predict(X), seg_gmm, meancov, '1,2,2', 'Gaussian Mixture')

        return repond_idx

    def find_pbbox(self,score_nms,pred_bbox_nms,score_size,scale_z):
        penalty = self.calc_penalty(pred_bbox_nms, scale_z)
        pscore = self.penalize_score(score_nms, penalty, score_size, True)
        best_idx = np.argmax(pscore)
        best_score = score_nms[best_idx]
        bbox = pred_bbox_nms[:, best_idx] / scale_z
        lr = penalty[best_idx] * score_nms[best_idx] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        pbbox = [cx, cy, width, height]

        return penalty,pbbox

    def find_proposal_bbox(self, penalty, score_nms,pred_bbox_nms, repond_idx, score_size, scale_z):

        proposal_bbox = []
        for cord in repond_idx:
            idx = np.ravel_multi_index(cord, (score_size, score_size))
            bb = pred_bbox_nms[:, idx] / scale_z

            lr = penalty[idx] * score_nms[idx] * cfg.TRACK.LR
            cx = bb[0] + self.center_pos[0]
            cy = bb[1] + self.center_pos[1]
            width = self.size[0] * (1 - lr) + bb[2] * lr
            height = self.size[1] * (1 - lr) + bb[3] * lr

            proposal_bbox.append([cx , cy , width, height])

        return proposal_bbox

    def merge_bbox(self,img,pbbox,proposal_bbox):

        def to_lurd(box):
            return [box[0] - box[2] / 2,box[1] - box[3] / 2,
                    box[0] + box[2] / 2,box[1] + box[3] / 2]

        filtered_bbox=[]
        for bb in proposal_bbox:
            iou_score=rubost_track.cal_iou(to_lurd(pbbox),to_lurd(bb))
            # print('iou: %f' %(iou_score))
            if iou_score <=0.7:
                cx, cy, width, height = self._bbox_clip(bb[0], bb[1], bb[2],
                                                        bb[3], img.shape[:2])
                bbox = [cx - width / 2,
                        cy - height / 2,
                        width,
                        height]
                filtered_bbox.append(bbox)

        return filtered_bbox

    def score_nms(self,outputs,score_size,anchors):
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)
        best_score = score[np.argmax(score)]

        index = np.argmax(score.reshape(5, -1), axis=0)
        score_nms = score.reshape(5, -1)[index, np.arange(score_size * score_size)]
        pred_bbox_nms = pred_bbox.reshape(4, 5, -1)[:, index, np.arange(score_size * score_size)]

        return score_nms,pred_bbox_nms,best_score

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

    def penalize_score(self, score, penalty, score_size, update_state):

        pscore = penalty * score

        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window=window.flatten()
        # window = np.tile(window.flatten(), self.anchor_num)

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
        score_nms,pred_bbox_nms,best_score=self.score_nms(outputs,score_size,anchors)
        score_map = score_nms.reshape(score_size, score_size)

        if self.visualize:
            cv2.namedWindow('Heated X_Crop', cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow('Heated X_Crop', 650, 220)
            x_crop = x_crop.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy().astype(np.uint8)
            frame_show = rubost_track.plot_xcrop_heated(x_crop, score_map, instance_size,
                                                        self.frame_num, best_score,
                                                        0, 0)
            cv2.imshow('Heated X_Crop', frame_show)
            cv2.waitKey(1)

        penalty, pbbox = self.find_pbbox(score_nms,pred_bbox_nms,score_size,scale_z)
        repond_idx = self.segment_groups(score_map)
        if not self.state_sampling:
            self.center_pos = np.array([pbbox[0], pbbox[1]])
            self.size = np.array([pbbox[2], pbbox[3]])

            cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                    pbbox[3], img.shape[:2])
            bbox = [cx - width / 2,
                    cy - height / 2,
                    width,
                    height]
            return {
            'bbox': bbox,
            's_x': s_x,
            'score_map': score_map}

        proposal_bbox = self.find_proposal_bbox(penalty, score_nms,pred_bbox_nms, repond_idx, score_size, scale_z)
        filtered_ppbbox = self.merge_bbox(img, pbbox, proposal_bbox)

        if self.state_update:
            self.template_upate(self.zf_gt, 0.0126)

        if self.state_lost:
            # compare hist with proposals
            pass
        if self.state_cross:
            self.cross_count+=1
            if self.cross_count < 5 :
                self.center_pos = np.array([pbbox[0], pbbox[1]])
                self.size = np.array([pbbox[2], pbbox[3]])

                cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                        pbbox[3], img.shape[:2])
                bbox = [cx - width / 2,
                        cy - height / 2,
                        width,
                        height]

                return {
                    'bbox': bbox,
                    's_x': s_x,
                    'score_map': score_map,
                    'proposal_bbox': filtered_ppbbox
                }


            if len(filtered_ppbbox):
                hist_score1,hist_score2 = self.similar_compare_deep(img,pbbox,filtered_ppbbox)
                hist_score=hist_score1/np.array(hist_score2)
                print(hist_score)
                if any(hist_score < 0.9):
                    idx=np.argmin(hist_score)
                    bbox=filtered_ppbbox[idx]

                    self.center_pos = np.array([bbox[0], bbox[1]])
                    self.size = np.array([bbox[2], bbox[3]])

                    self.state_cross=False
                else:
                    self.center_pos = np.array([pbbox[0], pbbox[1]])
                    self.size = np.array([pbbox[2], pbbox[3]])

                    cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                            pbbox[3], img.shape[:2])
                    bbox = [cx - width / 2,
                            cy - height / 2,
                            width,
                            height]
                    self.state_cross = False

                self.cross_count = 0

            else:
                self.center_pos = np.array([pbbox[0], pbbox[1]])
                self.size = np.array([pbbox[2], pbbox[3]])

                cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                        pbbox[3], img.shape[:2])
                bbox = [cx - width / 2,
                        cy - height / 2,
                        width,
                        height]


            return {
                'bbox': bbox,
                's_x': s_x,
                'score_map': score_map,
                'proposal_bbox': filtered_ppbbox
            }


        if best_score <= self.CONFIDENCE_LOW:
            self.state_update = False
            self.state_lost= True

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
        else:

            self.center_pos = np.array([pbbox[0], pbbox[1]])
            self.size = np.array([pbbox[2], pbbox[3]])

            cx, cy, width, height = self._bbox_clip(pbbox[0], pbbox[1], pbbox[2],
                                                    pbbox[3], img.shape[:2])
            bbox = [cx - width / 2,
                    cy - height / 2,
                    width,
                    height]

        if self.state_std:
            if not len(filtered_ppbbox):
                self.state_update = True

            self.last_patch = img[int(bbox[1]):int(bbox[1] + bbox[3]),
                     int(bbox[0]):int(bbox[0] + bbox[2]), :]
        else:
            self.state_update = False
            self.state_cross=True



        if self.state_update:
            self.template_upate(self.template(img, bbox), 0.0126)
            self.zf_bbox_global=self.feature_mix(self.zf_bbox_global,
                                                 self.img_similar.get_feature(self.crop_bbox(img,bbox)),
                                                 0.0126)

        if self.visualize:
            strshow = 'std:' + str(self.center_std).split('.')[0] + '.' + str(self.center_std).split('.')[1][:3]
            frame_show = cv2.putText(frame_show, strshow, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (0, 0, 255), 1, cv2.LINE_AA)
            strshow = 'update:' + str(self.state_update)
            frame_show = cv2.putText(frame_show, strshow, (110, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Heated X_Crop', frame_show)
            cv2.waitKey(1)

        return {
            'bbox': bbox,
            's_x': s_x,
            'score_map': score_map,
            'proposal_bbox': filtered_ppbbox
        }

    def compare_hist(self,img,bbox,filtered_ppbbox):
        patch1 = img[int(bbox[1]):int(bbox[1] + bbox[3]),
                 int(bbox[0]):int(bbox[0] + bbox[2]), :]

        patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB)
        hist1 = cv2.calcHist([patch1], [0, 1, 2], None, [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist_score1 = cv2.compareHist(self.hist_gt, hist1, cv2.HISTCMP_CORREL)

        hist_score2 = []

        for fbbox in filtered_ppbbox:
            fbbox=np.array(fbbox)
            fbbox[np.where(fbbox < 0 )]=0
            patch2 = img[int(fbbox[1]):int(fbbox[1] + fbbox[3]),
                     int(fbbox[0]):int(fbbox[0] + fbbox[2]), :]
            patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
            hist2 = cv2.calcHist([patch2], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()
            hist_score = cv2.compareHist(self.hist_gt, hist2, cv2.HISTCMP_CORREL)
            hist_score2.append(hist_score)

        # cv2.imshow('', patch2)
        # cv2.waitKey(1)

        return hist_score1,hist_score2

    def similar_compare_deep(self,img,bbox,filtered_ppbbox):
        score1 = self.img_similar.similarity_score(self.zf_bbox_global,
                                                   self.img_similar.get_feature(
                                                       self.crop_bbox(img,bbox)))

        score2 = []
        for fbbox in filtered_ppbbox:
            fbbox=np.array(fbbox)
            fbbox[np.where(fbbox < 0 )]=0
            patch2 = self.crop_bbox(img,fbbox)
            score = self.img_similar.cal_similarity_score(self.zf_bbox_global,
                                                          self.img_similar.get_feature(
                                                              self.crop_bbox(img, fbbox)))
            score2.append(score)

        # cv2.imshow('', patch2)
        # cv2.waitKey(1)

        return score1,score2

    def crop_bbox(self,img,bbox):
        return img[int(bbox[1]):int(bbox[1] + bbox[3]),
                 int(bbox[0]):int(bbox[0] + bbox[2]), :]

    def feature_mix(self,feature1,feature2,wt):
        return feature1*(1-wt) + feature2*wt