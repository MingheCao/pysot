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

    respond = score.reshape(5, 25, 25).max(0)
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