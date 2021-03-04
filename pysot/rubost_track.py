import random
from sklearn.mixture import GaussianMixture
import numpy as np

import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

def scoremap_sample(score):

    thre=0.2
    sig=0.15

    score_size=int(np.sqrt(score.size/5))

    respond = score.reshape(5, score_size, score_size).max(0)
    idx = np.where(respond >= thre)
    points=np.array(idx).transpose()

    zval=10*respond[idx]
    zval=zval.round().astype(np.int)
    point_set=points.repeat(zval,axis=0)

    gauss_rand=np.random.normal([0,0], sig, size=(len(point_set),2))
    point_set=point_set+gauss_rand

    return point_set,points

def gmm_fit(X):
    gm = GaussianMixture(n_components=4, covariance_type='full',random_state=0).fit(X)
    return gm

def KLdiv_gmm(gmm_p,gmm_q):
    res=0.
    for i in range(gmm_p.n_components):
        mu_p=gmm_p.means_[i]
        cov_p=gmm_p.covariances_[i]
        w_p=gmm_p.weights_[i]
        min_div = float("inf")

        for j in range(gmm_q.n_components):
            mu_q = gmm_q.means_[i]
            cov_q = gmm_q.covariances_[i]
            w_q = gmm_q.weights_[i]

            div=KLdiv_gm((mu_p,cov_p),(mu_q,cov_q))

            if div+np.log(w_p/w_q) <min_div:
                min_div=div+np.log(w_p/w_q)

        res+=w_p*min_div

    return res

# p = (mu1, Sigma1) = np.transpose(np.array([0.2, 0.1, 0.5, 0.4])), np.diag([0.14, 0.52, 0.2, 0.4])
# q = (mu2, Sigma2) = np.transpose(np.array([0.3, 0.6, -0.5, -0.8])), np.diag([0.24, 0.02, 0.31, 0.51])
def KLdiv_gm(p, q):
    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, title):
    splot = plt.subplot(2, 2, 4)
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

def plot(score):
    X, _ = scoremap_sample(score)
    X[:, [1, 0]] = X[:, [0, 1]]
    gm=gmm_fit(X)
    plot_results(X, gm.predict(X), gm.means_, gm.covariances_, 'Gaussian Mixture')

if __name__ == '__main__':
    score=np.load('/home/rislab/Workspace/pysot/tools/aa.npy')
    plot(score)
    plt.show()