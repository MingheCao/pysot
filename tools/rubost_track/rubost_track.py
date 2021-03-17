from sklearn.mixture import GaussianMixture
import numpy as np

import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from sklearn.cluster import DBSCAN

from tools.rubost_track.pycw import ChineseWhispers

import cv2
# def scoremap_sample(score):
#
#     thre=0.2
#     sig=0.15
#
#     score_size=int(np.sqrt(score.size/5))
#
#     respond = score.reshape(5, score_size, score_size).max(0)
#     idx = np.where(respond >= thre)
#     points=np.array(idx).transpose()
#
#     zval=10*respond[idx]
#     zval=zval.round().astype(np.int)
#     point_set=points.repeat(zval,axis=0)
#
#     gauss_rand=np.random.normal([0,0], sig, size=(len(point_set),2))
#     point_set=point_set+gauss_rand
#
#     return point_set,points

def scoremap_sample_reject(score_map,n_samples):
    score_size=score_map.shape[0]
    respond = score_map

    xy=np.random.rand(n_samples,2)*(score_size - 1)
    xy_cord=np.round(xy).astype(np.int32)
    z=respond[(xy_cord[:,0],xy_cord[:,1])]

    s = np.random.uniform(size=n_samples)
    idx=np.where(z>s)

    sample_points=xy[idx]
    sample_points[:,[0,1]]=sample_points[:,[1,0]] - score_size/2

    return sample_points


def gmm_fit(X,n_components):
    gm = GaussianMixture(n_components=n_components, covariance_type='full',random_state=0).fit(X)
    return gm

def KLdiv_gmm(gmm_p,gmm_q):
    res=0.
    for i in range(gmm_p['n_components']):
        mu_p=gmm_p['means_'][i]
        cov_p=gmm_p['covariances_'][i]
        w_p=gmm_p['weights_'][i]
        min_div = float("inf")

        for j in range(gmm_q['n_components']):
            mu_q = gmm_q['means_'][j]
            cov_q = gmm_q['covariances_'][j]
            w_q = gmm_q['weights_'][j]

            div=KLdiv_gm((mu_p,cov_p),(mu_q,cov_q))

            if div+np.log(w_p/w_q) <min_div:
                min_div=div+np.log(w_p/w_q)

        res+=w_p*min_div

    return res

def KLdiv_gmm_index(gmm_p,gmm_q,index):
    gmm1={
        'means_':gmm_p.means_[index],
        'covariances_':gmm_p.covariances_[index],
        'weights_':gmm_p.weights_[index],
        'n_components':len(index[0])
    }
    gmm2={
        'means_':gmm_q.means_,
        'covariances_':gmm_q.covariances_,
        'weights_':gmm_q.weights_,
        'n_components':gmm_q.n_components
    }

    kldiv=(KLdiv_gmm(gmm1,gmm2)+KLdiv_gmm(gmm2,gmm1))/2
    return kldiv

# p = (mu1, Sigma1) = np.transpose(np.array([0.2, 0.1, 0.5, 0.4])), np.diag([0.14, 0.52, 0.2, 0.4])
# q = (mu2, Sigma2) = np.transpose(np.array([0.3, 0.6, -0.5, -0.8])), np.diag([0.24, 0.02, 0.31, 0.51])
def KLdiv_gm(p, q):
    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)

def KLdiv_gm_weighted(p,q,w_p,w_q):
    kldiv=KLdiv_gm(p,q)
    kldiv=w_p*(kldiv+np.log(w_p/w_q))
    return kldiv

def construct_adjacency_mat(gmm,threshold):
    n_samples = gmm.n_components
    distances_mat=np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        mu_p = gmm.means_[i]
        cov_p = gmm.covariances_[i]
        w_p = gmm.weights_[i]
        for j in range(n_samples):
            if i==j:
                continue

            mu_q = gmm.means_[j]
            cov_q = gmm.covariances_[j]
            w_q = gmm.weights_[j]

            kldiv=(KLdiv_gm_weighted((mu_p,cov_p),(mu_q,cov_q),w_p,w_q) +
                  KLdiv_gm_weighted((mu_q, cov_q), (mu_p, cov_p), w_q, w_p))/2
            distances_mat[i,j]=kldiv

    adjacency_mat = (1 / (distances_mat + np.identity(n_samples, dtype=np.float64))) * \
                    (np.ones((n_samples, n_samples), dtype=np.float64) -
                     np.identity(n_samples, dtype=np.float64))

    adjacency_mat[np.where(adjacency_mat <= 1 / threshold)] = 0.
    return adjacency_mat

def ChineseWhispers_gm(gmm,threhold = 2,n_iter=3):

    adjacency_mat=construct_adjacency_mat(gmm,threhold)

    cw = ChineseWhispers(n_iteration=n_iter, metric='euclidean')
    predicted_labels = cw.fit_predict_gm(gmm.means_,adjacency_mat)
    return  predicted_labels


color_iter = itertools.cycle(['r','navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange','g'])

def plot_results(X, Y_, means, covariances, subplots,title):
    splot = plt.subplot(int(subplots.split(',')[0]), int(subplots.split(',')[1]), int(subplots.split(',')[2]))
    splot.cla()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(0., 25)
    plt.ylim(25, 0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def plot_results_cw(X, Y_, means, covariances, gmm_labels, center_label,subplots,title,gms_mean):
    splot = plt.subplot(int(subplots.split(',')[0]), int(subplots.split(',')[1]), int(subplots.split(',')[2]))
    splot.cla()
    label=np.unique(gmm_labels)

    for meancov in gms_mean:
        plt.plot(meancov[0][0],meancov[1][1],'X',color='b')

    for j in range(len(label)):
        idx=np.where(gmm_labels==label[j])
        means_=means[idx]
        covariances_=covariances[idx]
        if label[j]==center_label:
            color='r'
        else:
            color=next(color_iter)

        for i, (mean, covar,index) in enumerate(zip(
                means_, covariances_,idx[0])):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == index):
                continue
            plt.scatter(X[Y_ == index, 0], X[Y_ == index, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

    plt.xlim(-12.5, 12.5)
    plt.ylim(-12.5, 12.5)
    # plt.xticks(())
    # plt.yticks(())
    # plt.title(title)
    splot.xaxis.set_ticks_position('top')
    splot.invert_yaxis()


def visualize_response3d(outputs,fig,subplots,frame_num):

    score_map=outputs['score_map']
    if score_map.shape[0] != score_map.shape[1]:
        raise ValueError("width and height not equal.")
    score_size=score_map.shape[0]

    X = np.arange(0, score_size, 1)
    Y = np.arange(score_size, 0, -1)
    X, Y = np.meshgrid(X, Y)

    ax1 = plt.subplot(int(subplots.split(',')[0]), int(subplots.split(',')[1]), int(subplots.split(',')[2]),projection='3d')
    # ax1 = plt.subplot(121, projection='3d')
    ax1.cla()
    surf = ax1.plot_surface(X, Y, score_map, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_zlim(0.0, 1.0)
    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.invert_yaxis()
    # ax1.zaxis.set_major_locator(LinearLocator(5))
    # ax1.zaxis.set_major_formatter('{x:.01f}')
    # fig.colorbar(surf, shrink=0.4, aspect=5)

    # ax1.view_init(60, 90)
    plt.xticks([])
    plt.yticks([])
    ##

    # plt.savefig('/home/rislab/Workspace/pysot/rb_result/templates/respond_3d.png', dpi=300, bbox_inches='tight')

    plt.pause(0.1)

def add_text_info(outputs,frame_num):
    score_map=outputs['score_map']
    if score_map.shape[0] != score_map.shape[1]:
        raise ValueError("width and height not equal.")

    instance_size=outputs['instance_size']
    # respond = score.reshape(5, score_size, score_size).max(0)
    respond=score_map

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

def plot_search_area(outputs,frame):
    rect_length = outputs['s_x']
    bbox = list(map(int, outputs['bbox']))
    cor_x = int(bbox[0] + bbox[2] / 2 - rect_length / 2)
    cor_y = int(bbox[1] + bbox[3] / 2 - rect_length / 2)
    cv2.rectangle(frame, (cor_x, cor_y),
                  (int(cor_x + rect_length), int(cor_y + rect_length)),
                  (0, 0, 255), 3)
    return frame

def visualze_template(template,num):
    temp=template.permute(2, 3, 1, 0).squeeze().cpu().detach().numpy()

    heatmap=temp[:,:,1]
    heatmap=cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                  cv2.CV_8U)

    heatmap=np.repeat(heatmap,10,axis=0)
    heatmap = np.repeat(heatmap, 10, axis=1)

    heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_WINTER)

    cv2.imshow('', heatmap)
    cv2.waitKey(0)
    cv2.imwrite('/home/rislab/Workspace/pysot/rb_result/templates/'+ str(num) +'.jpg', heatmap)

def dbscan_clustering(X,eps,min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db

def plot_db(db):
    db=dbscan_clustering(X,)


if __name__ == '__main__':
    score=np.load('/home/rislab/Workspace/pysot/tools/aa.npy')
    plot(score)
    plt.show()