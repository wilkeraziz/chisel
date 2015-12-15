__author__ = 'waziz'

import numpy as np
import sys
from collections import defaultdict
from chisel.smt import groupby
from chisel.util import obj2id
from chisel.smt import Solution
from chisel.util import npvec2str, fmap_dot


def py(derivations, q_wmap, p_wmap, get_yield, empirical_q=True, alpha=1.0, beta=1.0):
    """
    :param support: list of DerivationGroup objects, each of which represents derivations sharing the same yield (Dy)
    :param p_features: list of features of the target
    :param q_features: list of features of the proxy
    :param get_yield: a function that returns the yield of a derivation
    :param alpha: p's dot scaling
    :param beta: q's dot scaling
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
    
    # these are the indices of the derivations projecting onto a certain string y
    y2D = [[] for _ in xrange(len(Y))]
    for d, y in enumerate(d2y):
        y2D[y].append(d)
    # helper function which selects statistics (from a given array) associated with derivations for which gamma_y(d) == 1 for a given y
    select = lambda array, y: array[y2D[y]]

    # 1) dot products
    q_dot = np.array([fmap_dot(d.vector, q_wmap) * beta for d in derivations])
    p_dot = np.array([fmap_dot(d.vector, p_wmap) * alpha for d in derivations])
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
    log_urd = r_dot + np.log(nd)
    log_rd = log_urd - np.logaddexp.reduce(log_urd)
    rd = np.exp(log_rd)

    # 5) p(y) 
    # where log up(y) = \sum_{d in Dy} log ur(d)
    log_upy = np.array([np.logaddexp.reduce(select(log_urd, y)) for y in Y])
    log_py = log_upy - np.logaddexp.reduce(log_upy)
    py = np.exp(log_py)

    return support, py, qy

def minrisk(derivations, q_wmap, p_wmap, get_yield, empirical_q=True, alpha=1.0, beta=1.0):
    """
    :param support: list of DerivationGroup objects, each of which represents derivations sharing the same yield (Dy)
    :param p_features: list of features of the target
    :param q_features: list of features of the proxy
    :param get_yield: a function that returns the yield of a derivation
    :param alpha: p's dot scaling
    :param beta: q's dot scaling
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
    
    # these are the indices of the derivations projecting onto a certain string y
    y2D = [[] for _ in xrange(len(Y))]
    for d, y in enumerate(d2y):
        y2D[y].append(d)
    # helper function which selects statistics (from a given array) associated with derivations for which gamma_y(d) == 1 for a given y
    select = lambda array, y: array[y2D[y]]

    # 1) dot products
    q_dot = np.array([fmap_dot(d.vector, q_wmap) * beta for d in derivations])
    p_dot = np.array([fmap_dot(d.vector, p_wmap) * alpha for d in derivations])
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
    log_urd = r_dot + np.log(nd)
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
    fdpd = fd * rd[:,np.newaxis]
    fdpd_y = fd * rd_y[:,np.newaxis] 
    # <f(d)>_p
    p_expected_f = fdpd.sum(0)
    # <\gamma_y(d) f(d)>_p
    p_expected_f_y = np.array([select(fdpd_y, y).sum(0) for y in Y])
    dpdt = alpha * (p_expected_f_y - p_expected_f) * py[:,np.newaxis]
    return support, py, dpdt

def maxelb(derivations, q_wmap, p_wmap, empirical_q=False):
    # counts
    nd = np.array([d.count for d in derivations], float)
    # dot products
    q_dot = np.array([fmap_dot(d.vector, q_wmap) for d in derivations])
    p_dot = np.array([fmap_dot(d.vector, p_wmap) for d in derivations])

    # posterior
    if empirical_q:
        Zn = nd.sum()
        qd = nd / Zn  # simply, the empirical distribution
        log_qd = np.log(qd)
    else:
        log_uqd = np.log(nd) + q_dot
        log_qd = log_uqd - np.logaddexp.reduce(log_uqd)
        qd = np.exp(log_qd)

    # expected features
    gd = np.array([d.vector.as_array(q_wmap.features) for d in derivations])
    q_expected_g = (gd * qd[:,np.newaxis]).sum(0)
    
    # Evidence lower bound
    #  <log ~p(d)>_q - <log q(d)>_q
    #  = < theta * f(d) >_q - <log q(d)>_q
    ELB = ((p_dot - log_qd) * qd).sum()
    dqdl = (gd - q_expected_g) * qd[:,np.newaxis]
    dELB = (dqdl * (p_dot - log_qd - 1)[:, np.newaxis]).sum(0)
    return ELB, dELB

def minkl(derivations, q_wmap, p_wmap, empirical_q=False):
    # counts
    nd = np.array([d.count for d in derivations], float)
    # dot products
    q_dot = np.array([fmap_dot(d.vector, q_wmap) for d in derivations])
    p_dot = np.array([fmap_dot(d.vector, p_wmap) for d in derivations])
    r_dot = p_dot - q_dot

    # posterior
    if empirical_q:
        Zn = nd.sum()
        qd = nd / Zn  # simply, the empirical distribution
        log_qd = np.log(qd)
    else:
        log_uqd = np.log(nd) + q_dot
        log_qd = log_uqd - np.logaddexp.reduce(log_uqd)
        qd = np.exp(log_qd)
    
    # importance weight: r(d) = ur(d)/Zr
    log_urd = r_dot + np.log(nd)
    log_rd = log_urd - np.logaddexp.reduce(log_urd)

    # expected features
    gd = np.array([d.vector.as_array(q_wmap.features) for d in derivations])
    q_expected_g = (gd * qd[:,np.newaxis]).sum(0)
    
    KL = (qd * (log_qd - log_rd)).sum()
    dKL = (((gd - q_expected_g).transpose() * qd) * (log_qd - log_rd + 1)).transpose().sum(0)
    return KL, dKL

def minvar(derivations, q_wmap, p_wmap, empirical_q=False, alpha=1.0, beta=1.0):
    pass

