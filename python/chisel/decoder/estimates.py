__author__ = 'waziz'

from chisel.smt import Solution
from chisel.util import npvec2str, fmap_dot
import numpy as np
import sys
from chisel.smt import groupby
from chisel.util import obj2id
from collections import defaultdict

class EmpiricalDistribution(object):
    """
    """

    def __init__(self, 
            derivations, 
            q_wmap,  # q_features
            p_wmap,  # p_features
            get_yield,
            empirical_q=True
            ):
        """
        :param support: list of DerivationGroup objects, each of which represents derivations sharing the same yield (Dy)
        :param p_features: list of features of the target
        :param q_features: list of features of the proxy
        :param get_yield: a function that returns the yield of a derivation
        """
        
        # 0) organise the support

        # group derivations by yield
        support = groupby(derivations, key=get_yield)
        # assign sequential ids to yields
        y2i = defaultdict()
        [obj2id(Dy.projection, y2i) for Dy in support]
        # support of derivations
        D = np.arange(len(derivations))
        # support of strings
        Y = np.arange(len(y2i))
        # map derivation id to yield id
        d2y = np.array([y2i[get_yield(d)] for d in derivations], int)
        # matrix which reproduces the indicator function \gamma_y(d)
        gamma = np.zeros((len(D), len(Y)), int)
        for i, y in enumerate(d2y):
            gamma[i,y] = 1
        # helper function which selects statistics (from a given array) associated with derivations for which gamma_y(d) == 1 for a given y
        select = lambda array, y: array[gamma[:,y].nonzero()[0]]

        # 1) dot products
        q_dot = np.array([fmap_dot(d.vector, q_wmap) for d in derivations])
        p_dot = np.array([fmap_dot(d.vector, p_wmap) for d in derivations])
        r_dot = p_dot - q_dot

        # 2) counts: n(d) and n(y) 
        nd = np.array([d.count for d in derivations], float)
        ny = np.array([select(nd, y).sum() for y in Y])

        # 3) instrumental probability: q(d) and q(y)
        if empirical_q:
            Zn = nd.sum()
            qd = nd / Zn  # simply, the empirical distribution
            log_qd = np.log(qd)
            qy = ny / Zn
        else:
            log_uqd = np.log(nd) + q_dot
            log_qd = log_uqd - np.logaddexp.reduce(log_uqd)
            qd = np.exp(log_qd)
            qy = np.array([select(qd, y).sum() for y in Y])
       
        # 4) importance weight: r(d) = ur(d)/Zr
        log_urd = np.log(nd) + r_dot
        log_rd = log_urd - np.logaddexp.reduce(log_urd)
        rd = np.exp(log_rd)

        # 5) p(y) 
        # where log up(y) = \sum_{d in Dy} log ur(d)
        log_upy = np.array([np.logaddexp.reduce(select(log_urd, y)) for y in Y])
        log_py = log_upy - np.logaddexp.reduce(log_upy)
        py = np.exp(log_py)

        # 6) r_y(d) = ur(d)/sum_Dy ur(d)
        log_rd_y = [log_urd[d] - log_upy[d2y[d]] for d in D] 
        rd_y = np.exp(log_rd_y)
       
        # 7) expected feature vectors 
        fd = np.array([d.vector.as_array(p_wmap.features) for d in derivations])
        fdpd = (fd.transpose() * rd).transpose()
        fdpd_y = (fd.transpose() * rd_y).transpose() 
        # <f(d)>_p
        p_expected_f = fdpd.sum(0)
        # <\gamma_y(d) f(d)>_p
        p_expected_f_y = np.array([select(fdpd_y, y).sum(0) for y in Y])
        dpdt = ((p_expected_f_y - p_expected_f).transpose() * py).transpose()
        
        gd = np.array([d.vector.as_array(q_wmap.features) for d in derivations])
        gdqd = (gd.transpose() * qd).transpose()
        # <g(d)>_q
        q_expected_g = gdqd.sum(0)
        
        # 8) KL(q||up) where up(d) = exp(theta f(d)) = exp(p_dot(d))
        #       = \sum_d q(d) log (q(d)/up(d))
        #       = \sum_d q(d) (log q(d) - log up(d))
        #       = \sum_d q(d) (log q(d) - p_dot(d))
        KL = (qd * (log_qd - log_rd)).sum()
        # dKL/dlambda = \sum_d q(d)(g(d) - <g(d)>_q)(log q(d) - log up(d) + 1)
        dKLdl = (((gd - q_expected_g).transpose() * qd) * (log_qd - log_rd + 1)).transpose().sum(0)

        # 9) store data
        self.support_ = support
        self.p_wmap_ = p_wmap
        self.q_wmap_ = q_wmap
        self.ny_ = ny
        self.qy_ = qy
        self.py_ = py
        self.dpdt_ = dpdt
        self.kl_ = KL
        self.dkldl_ = dKLdl


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
    def p_wmap(self):
        return self.p_wmap_

    @property
    def q_wmap(self):
        return self.q_wmap_

    def n(self, i):
        """
        Absolute counts of the i-the derivation group (0-based).
        Note that in the case of importance sampling, this is only meaningful wrt the instrumental distribution,
        in which case the normalised version represents the posterior q(y).
        """
        return self.ny_[i]

    def q(self, i):
        """a synonym for n(i)"""
        return self.qy_[i]

    def p(self, i):
        """
        Posterior of the i-th derivation group (0-based).
        That is, p(y) where support[i] = Dy = {d \in D: yield(d) = y}."""
        return self.py_[i]

    def copy_posterior(self):
        return self.py_.copy()

    def copy_dpdt(self):
        return self.dpdt_.copy()

    def kl(self):
        return self.kl_, self.dkldl_

    def __str__(self):
        strs = ['#p(y)\t#q(y)\t#y']
        for i, Dy in enumerate(self.support_):
            strs.append('{0}\t{1}\t{2}'.format(self.p(i),
                self.q(i),
                Dy.projection))
        return '\n'.join(strs)

    def solution(self, i):
        return Solution(Dy=self.support_[i],
                        p=self.p(i),
                        q=self.q(i))
