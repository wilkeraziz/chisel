__author__ = 'waziz'

from chisel.smt import Solution
from chisel.converter import npvec2str
import numpy as np


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

        # raw counts
        self.counts_ = np.array([Dy.count() for Dy in support])
        # normalising constant for counts (number of samples)
        self.Zn_ = float(self.counts_.sum(0))

        # up(y) = \sum_{d \in D_y} ur(d) * n(d)
        self.uYR = np.array([sum(d.importance * d.count for d in Dy) for Dy in support])

        # Z_p = \sum_y up(y)
        # normalising constant for importance weights
        self.Zp_ = self.uYR.sum(0)

        # p(y) = \frac{up(y)}{Z_p}
        self.Py = self.uYR/self.Zp_

        # expected feature vector (F) given yield(d) = y
        # Z <f_y> = \sum_{d \in Dy} gamma_y(d) f(d) ur(d) n(d)
        self.uYFR = np.array([reduce(sum, (d.vector.as_array(p_features) * d.ur * d.n for d in Dy)) for Dy in support])
        # <f_y>
        self.uFy_ = np.array([self.uYFR[i]/self.uYR[i] for i, Dy in enumerate(support)])

        # expected feature vector (F)
        # Z <f> = \sum_y <f_y>
        self.uFR = self.uYFR.sum(0)
        # <f>
        self.uf_ = self.uFR / self.Zp_

        # expected feature vector (G) given yield(d) = y
        # Z <g_y> = \sum_{d \in Dy} gamma_y(d) g(d) ur(d) n(d)
        self.uYGR = np.array([reduce(sum, (d.vector.as_array(q_features) * d.ur * d.n for d in Dy)) for Dy in support])
        # <g_y>
        self.uGy_ = np.array([self.uYGR[i]/self.uYR[i] for i, Dy in enumerate(support)])

        # expected feature vector (G)
        # Z <g> = \sum_y <g_y>
        self.uGR = self.uYGR.sum(0)
        # <g>
        self.ug_ = self.uGR / self.Zp_

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
        return self.counts_[i] if not normalise else self.counts_[i]/self.Zn_

    def p(self, i, normalise=True):
        """
        Posterior of the i-th derivation group (0-based).
        That is, p(y) where support[i] = Dy = {d \in D: yield(d) = y}."""
        return self.uYR[i] if not normalise else self.Py[i]

    def f(self, i):
        """f(y) where groups[i] = Dy"""
        return self.uFy_[i]  # self.uYFR[i]/self.uYR[i]

    def g(self, i=None):
        """g(y) where groups[i] = Dy"""
        return self.uGy_[i]  # self.uYGR[i]/self.uYR[i]

    def copy_posterior(self, normalise=True):
        return self.uYR.copy() if not normalise else self.Py

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
