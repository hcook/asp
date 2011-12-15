import unittest
import matplotlib as mpl
import pylab as pl
import itertools
import sys
import math
import timeit
import copy

from gmm import *

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 3]),
        ]
    return Y.astype(np.float32)

class EMTester(object):

    def __init__(self, from_file, num_subps):
        
        self.results = {}
        self.num_subplots = num_subps
        self.plot_id = num_subps*100 + 11
        self.from_file = from_file
        
        if from_file:
            self.X = np.ndfromtxt('IS1000a.csv', delimiter=',', dtype=np.float32)
            self.N = self.X.shape[0]
            self.D = self.X.shape[1]
        else:
            self.D = 2
            self.N = 600
            self.X = generate_synthetic_data(self.N)

    def new_gmm(self, M):
        self.M = M
        self.gmm = GMM(self.M, self.D, cvtype='diag')

    def test_pure_python(self):
        means, covars = self.gmm.train_using_python(self.X)
        if not self.from_file:
            Y = self.gmm.predict_using_python(self.X)
            self.results['Pure'] = (str(self.plot_id), means, covars, Y.T)
            self.plot_id += 1

    def test_sejits(self):        
        likelihood = self.gmm.train(self.X)
        if not self.from_file:
            means = self.gmm.components.means.reshape((self.gmm.M, self.gmm.D))
            covars = self.gmm.components.covars.reshape((self.gmm.M, self.gmm.D, self.gmm.D))
            Y = self.gmm.predict(self.X)
            if(self.plot_id % 10 <= self.num_subplots):
                self.results['_'.join(['ASP v',str(self.plot_id-(100*self.num_subplots+11)),'@',str(self.D),str(self.M),str(self.N)])] = (str(self.plot_id), copy.deepcopy(means), copy.deepcopy(covars), copy.deepcopy(Y))
                self.plot_id += 1
        return likelihood
        
    def plot(self):
        if self.from_file: return
        for t, r in self.results.iteritems():
            splot = pl.subplot(r[0], title=t)
            color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
            Y_ = r[3]
            for i, (mean, covar, color) in enumerate(zip(r[1], r[2], color_iter)):
                v, w = np.linalg.eigh(covar)
                u = w[0] / np.linalg.norm(w[0])
                pl.scatter(self.X.T[0,Y_==i], self.X.T[1,Y_==i], .8, color=color)
                angle = np.arctan(u[1]/u[0])
                angle = 180 * angle / np.pi
                ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)
        pl.show()
        pl.savefig('em_test')
        
if __name__ == '__main__':
    device_id = 0
    num_subplots = 4
    emt = EMTester(False, num_subplots)
    emt.new_gmm(3)
    emt.test_sejits()
    emt.test_sejits()
    emt.test_sejits()
    emt.test_sejits()
    emt.plot()
 