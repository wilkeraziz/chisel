__author__ = 'waziz'

from chisel.smt import Solution
from chisel.util import npvec2str
import numpy as np
import sys


class EmpiricalDistribution(object):
    """
    """

    def __init__(self, support, p_features, q_features):
        """
        :param support: list of DerivationGroup objects, each of which represents derivations sharing the same yield (Dy)
        :param p_features: list of features of the target
        :param q_features: list of features of the proxy
        """
        self.support_ = support
        self.q_features_ = q_features
        self.p_features_ = p_features

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
        
        # raw counts: the array self.counts_ is used to compute the function
        #   n(y) = \sum_d \gamma_y(d)
        self.counts_ = np.array([Dy.count() for Dy in support], float)
        # normalising constant for counts (number of samples)
        #   Zn = \sum_y n(y)
        self.Zn_ = self.counts_.sum(0)
        ####print >> sys.stderr, 'n(y)=', self.counts_

        # normalisation suggested in Murphy (2012)
        # seems a bit silly as it ends up dominated by q(d)
        ###self.unnorm_log_r_ = np.array([reduce(np.logaddexp, (d.log_importance for d in Dy)) for Dy in support], float)
        ###self.log_Zr_ = reduce(np.logaddexp, self.unnorm_log_r_)
        ###self.r_ = np.exp(self.unnorm_log_r_ - self.log_Zr_)
        ###print >> sys.stderr, 'R=', self.r_
        
        # normalised q(y) = n(y) / Zn
        self.Qy_ = self.counts_ / self.Zn_
        ###print >> sys.stderr, 'q(y)=', self.Qy_
        
        # unnormalised p(y)
        #   Zp * p(y) = \sum_d \gamma_y(d) * importance(d)
        # where importance(d) is the unnormalised importance weight of d, that is, (Zp*p(d))/(Zq*q(d))
        self.unnorm_p_ = np.array([np.sum(d.importance * d.count for d in Dy) for Dy in support], float)
        #HACK self.unnorm_p_ = np.array([self.r_[i] for i, Dy in enumerate(support)], float)
        ###print >> sys.stderr, 'unp(y)=', self.unnorm_p_
        # p's normalising constant
        self.Zp_ = self.unnorm_p_.sum(0)
        
        # p(y) = unnorm_p(y) / Zp
        self.Py_ = self.unnorm_p_ / self.Zp_
        ###print >> sys.stderr, 'p(y)=', self.Py_


        # unnormalised expected f(y) -- feature vector wrt the target distribution p
        #   Z <f_y> = \sum_{d \in Dy} gamma_y(d) f(d) ur(d) n(d)
        self.unnorm_f_ = np.array([reduce(sum, (d.vector.as_array(p_features) * d.importance * d.count for d in Dy)) for Dy in support], float)
        # normalised expected f(y)
        #   <f(y)> = unnorm_f(y)/unnorm_p(y)
        self.Fy_ = np.array([self.unnorm_f_[i]/self.unnorm_p_[i] for i, Dy in enumerate(support)])
        # expected feature vector <f(d)>
        self.uf_ = self.unnorm_f_.sum(0) / self.Zp_

        # unnoralised expected g(y) -- feature vector wrt the instrumental distribution q
        #   Z <g_y> = \sum_{d \in Dy} gamma_y(d) g(d) ur(d) n(d)
        self.unnorm_g_ = np.array([reduce(sum, (d.vector.as_array(q_features) * d.importance * d.count for d in Dy)) for Dy in support], float)
        # normalised expected g(y)
        self.Gy_ = np.array([self.unnorm_g_[i]/self.unnorm_p_[i] for i, Dy in enumerate(support)])
    #def qq(self, i, normalise=True):
    #    return self.unnormalised_q_[i] if not normalise else self.normalised_q_[i]

        # expected feature vector <g(d)>
        self.ug_ = self.unnorm_g_.sum(0) / self.Zp_

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

    def n(self, i, normalise=False):
        """
        Absolute counts of the i-the derivation group (0-based).
        Note that in the case of importance sampling, this is only meaningful wrt the instrumental distribution,
        in which case the normalised version represents the posterior q(y).
        """
        return self.counts_[i] if not normalise else self.counts_[i]/self.Zn_

    def q(self, i, normalise=True):
        """a synonym for n(i)"""
        return self.Qy_[i] if normalise else self.counts_[i]

    def p(self, i, normalise=True):
        """
        Posterior of the i-th derivation group (0-based).
        That is, p(y) where support[i] = Dy = {d \in D: yield(d) = y}."""
        return self.Py_[i] if normalise else self.unnorm_p_[i]

    def f(self, i):
        """f(y) where groups[i] = Dy"""
        return self.Fy_[i] 

    def g(self, i=None):
        """g(y) where groups[i] = Dy"""
        return self.Gy_[i] 

    def copy_posterior(self, normalise=True):
        return self.Py_.copy() if normalise else self.unnorm_p_

    def uf(self):
        """Expected feature vector <ff>"""
        return self.uf_

    def ug(self):
        """Expected feature vector <gg>"""
        return self.ug_

    def dtheta(self, i):
        """\derivative{p(y;theta)/q(y;lambda)}{theta} where groups[i] = Dy"""
        return (self.f(i) - self.uf()) * self.p(i, True)

    def dlambda(self, i):
        """\derivative{p(y;theta)/q(y;lambda)}{lambda} where groups[i] = Dy"""
        return (self.ug() - self.g(i)) * self.p(i, True)

    def __str__(self):
        strs = []
        for i, Dy in enumerate(self.support_):
            strs.append('p={0}\tf={1}\tg={2}\tyield={3}'.format(self.p(i),
                                                                npvec2str(self.f(i)),
                                                                npvec2str(self.g(i)),
                                                                Dy.projection))
        return '\n'.join(strs)

    def solution(self, i):
        return Solution(Dy=self.support_[i],
                        f=self.f(i),
                        g=self.g(i),
                        p=self.p(i, normalise=True),
                        q=self.q(i, normalise=True),
                        n=self.n(i, normalise=False))
