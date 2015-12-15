__author__ = 'waziz'

import numpy as np
import sys
from collections import defaultdict
from chisel.smt import Solution
from chisel.util import npvec2str
from chisel.smt import groupby
from chisel.util import obj2id


class EmpiricalDistribution(object):
    """
    """

    def __init__(self, 
            derivations, 
            proxy_fmap,  # q_features
            tartet_weights,  # p_features
            get_yield):
        """
        :param support: list of DerivationGroup objects, each of which represents derivations sharing the same yield (Dy)
        :param p_features: list of features of the target
        :param q_features: list of features of the proxy
        :param get_yield: a function that returns the yield of a derivation
        """
        self.support_ = groupby(derivations, key=get_yield)
        #self.q_features_ = q_features
        #self.p_features_ = p_features

        # each group D_y constains derivations d such that their projections are y
        # the posterior of y is given by
        #  p(y) = \frac{ \expec{gamma_y(d) ur(d)}{q} }{ \expec{ur(d)}{q} }
        # where gamma_y(d) = 1 if yield(d)=y, 0 otherwise
        #  ur(d) = up(d)/uq(d)
        #  and the expectations are taken wrt q(d)
        #
        # we can estimate the unnormalised posterior of y using the empirical average
        #  up(y) = \sum_{d \in D_y} ur(d) * n(d)
        # where n(d) is the number of times d was sampled
        # the normalised posterior is p(y) = up(y) / \sum_y up(y)

        # From here, each position in a vector represents a unique string y
        # the set of derivations that yield y is denoted Dy

        

        Y = defaultdict()
        # get sequential ids for the strings in the support
        [obj2id(Dy.projection, Y) for Dy in self.support_]

        # n(d): how many times d has been sampled
        nd = np.array([d.count for d in derivations], float)
        # q(d): instrumental probability
        if not resample_q:
            # n(d) / Zn
            qd = nd / nd.sum()  # simply, the empirical distribution
        else:
            # n(d)uq(d)/(\sum_d' n(d')uq(d'))
            log_uqd = np.array([np.log(d.count) + d.log_uq for d in derivations], float)
            log_qd = log_uqd - np.logaddexp.reduce(log_uqd)
            qd = np.exp(log_qd)
            # TODO: resample
       
        log_urd = np.array([np.log(d.count) + d.log_ur for d in derivations], float)
        log_rd = log_urd - np.loggaddexp.reduce(log_urd)
        rd = np.exp(log_rd)
        # TODO: resample


        # r_num(d) = n(d) * ur(d)
        r_num = np.log(D[:,0]) + D[:,1]
        # Zr = \sum_d r_num(d)
        Zr = np.logaddexp.reduce(r_num)
        # r(d) = r_num(d)/Zr
        log_r = r_num - Zr
        r = np.exp(log_r)
       
        

        # raw counts: the array self.counts_ is used to compute the function
        #   n(y) = \sum_d \gamma_y(d)
        self.counts_ = np.array([Dy.count() for Dy in self.support_], float)
        # normalising constant for counts (number of samples)
        #   Zn = \sum_y n(y)
        self.Zn_ = self.counts_.sum(0)

        # this array represents all sampled derivations
        D = np.array([np.array([d.count, d.log_ur, obj2id(get_yield(d), Y)]) for i, d in enumerate(derivations)])
        # this matrix reproduces the indicator function \gamma_y(d)
        gamma = np.zeros((len(D), len(Y)), int)
        for i, row in enumerate(D):
            gamma[i,row[2]] = 1
        select = lambda array, y: array[gamma[:,y].nonzero()[0]]

        # r_num(d) = n(d) * ur(d)
        r_num = np.log(D[:,0]) + D[:,1]
        # Zr = \sum_d r_num(d)
        Zr = np.logaddexp.reduce(r_num)
        # r(d) = r_num(d)/Zr
        log_r = r_num - Zr
        r = np.exp(log_r)

        # py_num(y) = \sum_{d in Dy} r_num(d)
        py_num = np.array([np.logaddexp.reduce(select(r_num, y)) for y in np.arange(len(Y))])
        # p(y) = py_num(y) / Zr
        log_py = py_num - Zr
        py = np.exp(log_py)

        # this is just a repetition of py_num(y) for d such that yield(d)=y 
        # ry_den(d) = sum_Dy n(d) ur(d)
        ry_den = np.array([py_num[D[i,2]] for i, d in enumerate(derivations)])
        # ry(d) = n(d) ur(d)/(sum_Dy n(d) ur(d))
        log_ry = r_num - ry_den
        ry = np.exp(log_ry)
       
        # computes expected feature vectors
        fvecs = np.array([d.vector.as_array(p_features) for i, d in enumerate(derivations)]) 
        fd = (fvecs.transpose() * r).transpose()
        fy = (fvecs.transpose() * ry).transpose()
        
        gvecs = np.array([d.vector.as_array(q_features) for i, d in enumerate(derivations)])
        gd = (gvecs.transpose() * r).transpose()
        gy = (gvecs.transpose() * ry).transpose()
        
        # stores arrays
        self.qy_ = self.counts_ / self.Zn_
        
        self.log_r_ = log_r
        self.r_ = r

        self.log_ry_ = log_ry
        self.ry_ = ry

        self.log_py_ = log_py
        self.py_ = py

        # expected f(d) in D
        self.mean_f_ = fd.sum(0)
        # expected f(d) in Dy
        self.fy_ = np.array([select(fy, y).sum(0) for y in np.arange(len(Y))])
        # dp(y)/dtheta
        self.dpdt_ = ((self.fy_ - self.mean_f_).transpose() * py).transpose()
        
        # expected g(d) in D
        self.mean_g_ = gd.sum(0)
        # expected g(d) in Dy
        self.gy_ = np.array([select(gy, y).sum(0) for y in np.arange(len(Y))])
        # dp(y)/dlambda
        self.dpdl_ = ((self.gy_ - self.mean_g_).transpose() * py).transpose()

        # KL 
        qd_ = D[:,0]/self.Zn_
        self.kl_from_q_ = - (qd_ * log_r).sum()
        self.dkl_from_q_ = (((gd - self.mean_g_).transpose() * qd_) * (1 - log_r)).transpose().sum(0)

        self.kl_from_p_ = (r * log_r).sum()
        self.dkl_from_p_ = ((self.mean_g_ - gd).transpose() * r).transpose().sum(0)

    def __iter__(self):
        return iter(self.support_)

    def __getitem__(self, i):
        return self.support_[i]

    def __len__(self):
        return len(self.support_)

    @property
    def support(self):
        return self.support_

    @property
    def p_features(self):
        return self.p_features_

    @property
    def q_features(self):
        return self.q_features_

    def n(self, i):
        """
        Absolute counts of the i-the derivation group (0-based).
        Note that in the case of importance sampling, this is only meaningful wrt the instrumental distribution,
        in which case the normalised version represents the posterior q(y).
        """
        return self.counts_[i]

    def q(self, i):
        """a synonym for n(i)"""
        return self.qy_[i]

    def p(self, i):
        """
        Posterior of the i-th derivation group (0-based).
        That is, p(y) where support[i] = Dy = {d \in D: yield(d) = y}."""
        return self.py_[i]

    def f(self, i):
        """f(y) where groups[i] = Dy"""
        return self.fy_[i] 
    
    def g(self, i):
        """g(y) where groups[i] = Dy"""
        return self.gy_[i] 

    def copy_posterior(self):
        return self.py_.copy()

    def copy_dtheta(self):
        return self.dpdt_.copy()

    def uf(self):
        """Expected feature vector <ff>"""
        return self.mean_f_
    
    def gf(self):
        """Expected feature vector <ff>"""
        return self.mean_g_

    def dtheta(self, i):
        """\derivative{p(y;theta)/q(y;lambda)}{theta} where groups[i] = Dy"""
        return (self.f(i) - self.uf()) * self.p(i)

    def kl(self):
        return self.kl_from_q_, self.dkl_from_q_

    def __str__(self):
        strs = ['#p\t#q\t#f\t#dp\t#y']
        for i, Dy in enumerate(self.support_):
            strs.append('{0}\t{1}\t{2}\t{3}\t{4}'.format(self.p(i),
                self.q(i),
                npvec2str(self.f(i)),
                npvec2str(self.dtheta(i)),
                Dy.projection))
        strs.append('#\t#\t#<f>\t#\t#')
        strs.append(' \t \t%s\t \t ' % npvec2str(self.uf()))
        return '\n'.join(strs)

    def solution(self, i):
        return Solution(Dy=self.support_[i],
                        f=self.f(i),
                        g=self.g(i),
                        p=self.p(i),
                        q=self.q(i),
                        n=self.n(i))
