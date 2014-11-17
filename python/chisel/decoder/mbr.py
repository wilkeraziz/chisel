"""
Minimum Bayes Risk decoding for SMT (Kumar and Byrne, 2003):

    The risk associated with a certain candidate prediction c is given by
        R(c) = \sum_r L_r(c)p(r)
    where L_r(c) is the loss incurred in predicting c when r is the reference
     and p(r) is the belief associated with r.

    The decision rule becomes:
        c^ = \argmin_c \sum_r L_r(c)p(r)

    c \in C
        where C is the space of candidates considered for selection (hypothesis space)
    r \in R
        where R is the space of references considered to compute risk (evidence space)

Empirical Risk Minimisation (Smith and Eisner, 2006):

    w^ = \argmin_w \sum_{(x,r) \in D} \sum_c L_r(c)p(c|x; w)

    where w are the parameters of a log-linear model
        p(c|x; w) \propto \exp(w * f(x, c))

    where D is a training set consisting of pairs (x,r)
        where x is an input and r is a reference translation (or set of reference translations)

    L_r(c) is the loss associated with c when r is the reference
    p(c|x;w) is the belief associated with c

@author waziz
"""
import numpy as np
import chisel.mteval as mteval


def MBR(empdist, metric, normalise=False):
    """
    :param empdist: for now MBR assumes Yh == Ye (see above)
    :param metric:
    :param normalise:
    :return:
    """
    M = len(empdist)
    scores = np.array([0.0] * M)
    for h, hyp in enumerate(empdist):
        for r, ref in enumerate(empdist):
            score = mteval.loss(c=h, r=r, metric=metric)
            scores[h] += score * empdist.p(r, normalise)
    return scores

