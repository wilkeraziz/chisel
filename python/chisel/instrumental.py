"""
@author waziz
"""
import logging
import sys
import argparse
import math
import os
import numpy as np
import traceback
from itertools import chain, groupby
from collections import namedtuple, Counter, defaultdict
from time import time
from multiprocessing import Pool
from functools import partial
from ConfigParser import RawConfigParser
from scipy.optimize import minimize, basinhopping
from tabulate import tabulate
from scipy import linalg as LA

import chisel.ff as ff
import chisel.cdeclib as cdeclib
from chisel.util import obj2id 
from chisel.util import fpairs2str, dict2str, fmap_dot, scaled_fmap
from chisel.util import resample as do_resample
from chisel.util.config import configure, section_literal_eval
from chisel.util.iotools import SegmentMetaData, smart_ropen, smart_wopen
from chisel.smt import SVector, Tree, Derivation
from chisel.util.logtools import timethis


class Sampler(object):
        
    RawSample = namedtuple('RawSample', 'projection vector count')

    def __init__(self, segment, proxy_weights, target_weights, cdec_config_str=''):
        """returns the proxy distribution encoded as a hypergraph"""
        # creates a decoder if necessary
        self.segment_ = segment
        self.cdec_config_str_ = cdec_config_str
        self.proxy_weights_ = proxy_weights
        self.target_weights_ = target_weights
        self._decoder = None # laxy computation
        self._forest = None  # lazy computation
        self._features = None  # lazy computation

    @property
    def segment(self):
        return self.segment_

    @property
    def decoder(self):
        if self._decoder is None:
            self._decoder = cdeclib.create_decoder(self.cdec_config_str_, self.proxy_weights_)
        return self._decoder

    @property
    def forest(self):
        """returns the hypergraph (produces it if necessary)"""
        if self._forest is None:
            logging.info('parsing (%d): %s', self.segment_.id, self.segment_.src)
            # builds the proxy distribution
            self._forest = cdeclib.build_proxy(str(self.segment_.src), self.segment_.grammar, self.decoder)
        return self._forest

    @property
    def features(self):
        """returns all features in the forest (gathers it from forest if necessary)"""
        if self._features is None:
            self._features = frozenset(chain(*((f for f, v in e.feature_values) for e in self.forest.edges)))
        return self._features

    @timethis('Time to reweight the forest')
    def reweight(self, proxy_weights):
        self.proxy_weights_ = proxy_weights
        if self._forest is not None:
            self.forest.reweight(cdeclib.make_sparse_vector(proxy_weights))

    @timethis('Time to sample')
    def sample(self, n_samples):
        """samples translation derivations from a forest"""
        # pre-process the input (some scorers might require analysis of the input segment)
        ff.preprocess_input(self.segment_)
        # samples from the proxy distribution
        q_samples = cdeclib.sample(self.forest, n_samples)
        raw_samples = []
        for derivation_str, sample_info in sorted(q_samples.iteritems(), key=lambda pair: len(pair[1]), reverse=True):
            # computes additional features
            extraff = ff.compute_features(ff.Hypothesis(source=str(self.segment_.src), translation=derivation_str))
            # groups vectors associated with equivalent derivations
            counter = Counter(fpairs for fpairs, _ in sample_info)
            # compute target vectors
            for q_fpairs, count in counter.iteritems():
                # start with the features that are used in the proxy
                fmap = dict(q_fpairs)
                # include additional features (must not overwrite proxy features)
                for fname, fvalue in extraff:
                    fmap[fname] = fvalue
                raw_samples.append(Sampler.RawSample(projection=derivation_str, vector=SVector(fpairs=fmap.iteritems()), count=count))  # TODO: get the actual derivation string from cdec
                    
        # resets scorers to a null state
        ff.reset_scorers()
        return raw_samples 

    def save(self, raw_samples, odir, suffix=''):
        with smart_wopen('{0}/{1}{2}.gz'.format(odir, self.segment_.id, suffix)) as fo:
            print >> fo, '[proxy]'
            print >> fo, '\n'.join('{0}={1}'.format(k, v) for k, v in sorted(self.proxy_weights_.iteritems(), key=lambda (k,v): k))
            print >> fo

            print >> fo, '[target]'
            print >> fo, '\n'.join('{0}={1}'.format(k, v) for k, v in sorted(self.target_weights_.iteritems(), key=lambda (k,v): k))
            print >> fo

            print >> fo, '[samples]'
            print >> fo, '# count projection vector'
            for sample in sorted(raw_samples, key=lambda r: r.count, reverse=True):
                print >> fo, '{0}\t{1}\t{2}'.format(sample.count, sample.projection, sample.vector)


class KLYEstimates(object):
    """
    """

    def __init__(self,
            derivations,
            q_wmap,  # q_features
            p_wmap,  # p_features
            active_features,
            empirical_q=True,
            normalise_p=False,
            stderr=None):
        """
        :param derivations: list of samples (Sampler.Sample)
        :param q_wmap: a dict-like object representing the non-zero parameters of the proxy distribution
        :param q_wmap: a dict-like object representing the non-zero parameters of the target distribution
        :param active_features: the (already ordered) list of active features
        :param empirical_q: if true, q(d) is estimated as n(d)/N (using the count alone),
            if false, q(d) is estimated as n(d)uq(d)/\sum {n(d')uq(d')} (using the current dot product to smooth the counts -- remember that uq(d) = exp(lambda * g(d)))
        :param normalise_p: if true, we estimate KL(q||p) where p(d) is estimated (via importance sampling),
            if false, we estimate KL(q||up) = KL(q||p) - log Zp
        """

        # group derivations by yield
        y2i = defaultdict()
        yields = []
        for projection, Dy in groupby(derivations, key=lambda d: d.projection):
            obj2id(projection, y2i)
            yields.append(projection)

        # support of strings
        Y = np.arange(len(y2i))
        # map derivation id to yield id
        d2y = np.array([y2i[d.projection] for d in derivations], int)

        # these are the indices of the derivations projecting onto a certain string y
        y2D = [[] for _ in xrange(len(Y))]
        for d, y in enumerate(d2y):
            y2D[y].append(d)
        # helper function which selects statistics (from a given array) associated with derivations for which gamma_y(d) == 1 for a given y
        select = lambda array, y: array[y2D[y]]

        # 1) dot products
        gd = np.array([fmap_dot(d.vector, q_wmap) for d in derivations])
        fd = np.array([fmap_dot(d.vector, p_wmap) for d in derivations])
        hd = fd - gd

        # 2) q(y) \propto \sum_{d \in D_y} n(d)
        nd = np.array([d.count for d in derivations], float)  # derivation counts (from Q)
        Zq = nd.sum(0)
        qd = nd / Zq
        log_qd = np.log(qd)
        ny = np.array([select(nd, y).sum() for y in Y])  # string counts (from Q)
        qy = ny / Zq
        log_qy = np.log(qy)

        # 3) r(d) \propto exp(h(d)) * n(d)
        log_urd = hd + np.log(nd)
        log_rd = log_urd - np.logaddexp.reduce(log_urd)
        rd = np.exp(log_rd)

        # 4) p(y) = \sum_{d \in D_y} r(d)
        log_py = np.array([np.logaddexp.reduce(select(log_rd, y)) for y in Y])
        py = np.exp(log_py)

        # 5) expected feature vector
        gvecs = np.array([d.vector.as_array(active_features) for d in derivations])
        posteriors = (gvecs.transpose() * qd).transpose()
        expected_g = posteriors.sum(0)

        # 6) KL(q(y)||p(y)) = <log(q(y)/p(y))>_q(y)
        KL = (qy * (log_qy - log_py)).sum()

        # 7) derivative of KL wrt the parameters of q
        dKL = (((gvecs - expected_g).transpose() * qd) * (log_qd - fd + 1)).transpose().sum(0)

        if stderr is not None:
            str_yields = np.array([y.decode('ascii', 'ignore') for y in yields])
            print >> stderr, tabulate(np.column_stack((qy, py, str_yields)), headers=['q(y)', 'p(y)', 'yield'])

        # 9) store data
        self.KL_ = KL
        self.dKL_ = dKL

        # 10) effective count

        log_urd2 = log_urd * 2  # r(d)^2
        log_rd2 = log_urd2 - np.logaddexp.reduce(log_urd2)
        rd2 = np.exp(log_rd2)
        mean_r = rd.mean()
        mean_sr = np.exp(log_rd * 2).mean()
        ne = rd.size * mean_r * mean_r / mean_sr
        self.ne_ = ne
        self.dne_ = ((gvecs.transpose() * (rd2 - rd)) * 2 * ne).transpose().sum(0)


    @property
    def KL(self):
        return self.KL_

    @property
    def dKL(self):
        return self.dKL_

    @property
    def ne(self):
        return self.ne_

    @property
    def dne(self):
        return self.dne_



class KLDEstimates(object):
    """
    """

    def __init__(self, 
            derivations, 
            q_wmap,  # q_features
            p_wmap,  # p_features
            active_features, 
            empirical_q=True,
            normalise_p=False,
            stderr=None):
        """
        :param derivations: list of samples (Sampler.Sample)
        :param q_wmap: a dict-like object representing the non-zero parameters of the proxy distribution
        :param q_wmap: a dict-like object representing the non-zero parameters of the target distribution
        :param active_features: the (already ordered) list of active features
        :param empirical_q: if true, q(d) is estimated as n(d)/N (using the count alone),
            if false, q(d) is estimated as n(d)uq(d)/\sum {n(d')uq(d')} (using the current dot product to smooth the counts -- remember that uq(d) = exp(lambda * g(d)))
        :param normalise_p: if true, we estimate KL(q||p) where p(d) is estimated (via importance sampling),
            if false, we estimate KL(q||up) = KL(q||p) - log Zp
        """

        # 1) dot products
        q_dot = np.array([fmap_dot(d.vector, q_wmap) for d in derivations])
        p_dot = np.array([fmap_dot(d.vector, p_wmap) for d in derivations])
        r_dot = p_dot - q_dot

        # 2) counts: n(d) and n(y) 
        nd = np.array([d.count for d in derivations], float)

        # 3) instrumental probability: q(d) and q(y)
        if empirical_q:
            Zn = nd.sum()
            qd = nd / Zn  # simply, the empirical distribution
            log_qd = np.log(qd)
        else:
            log_uqd = np.log(nd) + q_dot
            log_qd = log_uqd - np.logaddexp.reduce(log_uqd)
            qd = np.exp(log_qd)
       
        # 4) importance weight: r(d) = ur(d)/Zr
        log_urd = np.log(nd) + r_dot
        log_rd = log_urd - np.logaddexp.reduce(log_urd)
        rd = np.exp(log_rd)

        # 5) expected feature vectors 
        gd = np.array([d.vector.as_array(active_features) for d in derivations])
        gdqd = (gd.transpose() * qd).transpose()
        # <g(d)>_q
        q_expected_g = gdqd.sum(0)
        
        # 6) KL(q||up) where up(d) = exp(theta f(d)) = exp(p_dot(d))
        #       = \sum_d q(d) log (q(d)/up(d))
        #       = \sum_d q(d) (log q(d) - log up(d))
        #       = \sum_d q(d) (log q(d) - p_dot(d))
        if normalise_p:
            KL = (qd * (log_qd - log_rd)).sum()
            dKLdl = (((gd - q_expected_g).transpose() * qd) * (log_qd - log_rd + 1)).transpose().sum(0)
        else:
            KL = (qd * (log_qd - p_dot)).sum()
            dKLdl = (((gd - q_expected_g).transpose() * qd) * (log_qd - p_dot + 1)).transpose().sum(0)

        if stderr is not None:  # TODO report aggregated (by Y) values
            Y = np.array([d.projection.decode('ascii', 'ignore') for d in derivations])
            print >> stderr, tabulate(np.column_stack((qd, rd, log_qd - log_rd, log_qd - p_dot, p_dot - q_dot, Y)), headers=['q(d)', 'p(d)', 'log q(d) - log p(d)', 'log q(d) - log up(d)', 'up(d) - uq(d)', 'yield'])

        # 9) store data
        self.p_wmap_ = p_wmap
        self.q_wmap_ = q_wmap
        self.nd_ = nd
        self.qd_ = qd
        self.rd_ = rd
        self.KL_ = KL
        self.dKL_ = dKLdl


        # 10) effective count

        log_urd2 = log_urd * 2  # r(d)^2
        log_rd2 = log_urd2 - np.logaddexp.reduce(log_urd2)
        rd2 = np.exp(log_rd2)
        mean_r = rd.mean()
        mean_sr = np.exp(log_rd * 2).mean()
        ne = rd.size * mean_r * mean_r / mean_sr
        self.ne_ = ne
        self.dne_ = ((gd.transpose() * (rd2 - rd)) * 2 * ne).transpose().sum(0)

    @property
    def KL(self):
        return self.KL_

    @property
    def dKL(self):
        return self.dKL_

    @property
    def ne(self):
        return self.ne_

    @property
    def dne(self):
        return self.ne_

class KLOptimiser(object):
        
    DivergenceReturn = namedtuple('DivergenceReturn', 'KL dKL ne dne')
    NoRegularisation = None
    L1Regulariser = lambda w: LA.norm(w, 1)
    L2Regulariser = lambda w: LA.norm(w, 2)

    def __init__(self, 
            segment, 
            n_samples, 
            qmap, 
            pmap, 
            cdec_config_str, 
            regulariser='', #L1Regulariser, #NoRegularisation,
            regulariser_weight=1.0,
            ne_weight=0.0,
            avgcoeff=1.0):

        if not (0 < avgcoeff <= 1.0):
            raise ValueError('The coefficient of the moving average must be such that 0 < c <= 1.0, got %s' % avgcoeff)

        self.n_samples_ = n_samples
        self.qmap0_ = qmap
        self.pmap_ = pmap
        self.sampler_ = Sampler(segment, qmap, pmap, cdec_config_str)
        self.features_ = sorted(self.sampler_.features)
        logging.info('%d features', len(self.features_))
        self.f2i_ = defaultdict(None, ((f,i) for i, f in enumerate(self.features_)))
        self.regulariser_weight_ = regulariser_weight
        if regulariser.lower() == 'l1':
            self.regulariser_ = 1
        elif regulariser.lower() == 'l2':
            self.regulariser_ = 2
        else:
            self.regulariser_ = 0
        self.ne_weight_ = ne_weight
        logging.info('Q: %s', qmap)
        logging.info('P: %s', pmap)
        self.avgcoeff_ = avgcoeff
        self.sample_history_ = None

    def qmap0(self):
        return self.qmap0_

    def dict2nparray(self, wmap):
        """converts a sparse weight vector (as a dict) into a dense np array"""
        weights = np.zeros(len(self.features_))
        for f, w in wmap.iteritems():
            weights[self.f2i_.get(f, 0)] = w
        return weights

    def nparray2dict(self, array):
        """converts a dense weight vector (as a numpy array) into a sparse weight vector (as a dict)"""
        nids = array.nonzero()[0]
        return defaultdict(None, ((self.features_[fid], array[fid]) for fid in nids))

    def sample(self, qmap, n_samples=None):
        """returns samples from q grouped by derivation"""
        if n_samples is None:
            n_samples = self.n_samples_
        # reweights the forest
        self.sampler_.reweight(qmap)
        # samples
        samples = self.sampler_.sample(n_samples)
        # if the moving average coefficient of the current batch is 1.0, there is no point in computing considering previous samples
        if self.avgcoeff_ == 1.0:
            return samples
        # otherwise we might need to merge sampled batches
        if self.sample_history_ is None:  # except of course, in the first round
            self.sample_history_ = samples
            return samples
        # compute a moving average
        dmap = defaultdict(lambda : defaultdict(float))
        for d in samples:
            dmap[d.projection][d.vector] = d.count * self.avgcoeff_
        for d in self.sample_history_:
            dmap[d.projection][d.vector] += d.count * (1 - self.avgcoeff_)
        samples = []
        for projection, dinfo in dmap.iteritems():
            for vector, count in dinfo.iteritems():
                samples.append(Sampler.RawSample(projection=projection, vector=vector, count=count))
        self.sample_history_ = samples
        return samples


    def estimates(self, samples, qmap, pmap):
        """returns the objective and the jacobian"""
        # compute KL estimates
        # TODO: implement moving average of estimates
        # TODO: fix lambda (the part that is common between q and p)
        empdist = KLDEstimates(samples,
                q_wmap=qmap,
                p_wmap=pmap,
                active_features=self.features_,
                empirical_q=False,
                normalise_p=False,
                stderr=sys.stderr)
        return KLOptimiser.DivergenceReturn(empdist.KL, empdist.dKL, empdist.ne, empdist.dne)

    def optimise(self):
        """optimises the proxy for min KL(q||p)"""
       
        self.step = 1
        self.iteration = 1
        self.funcall = 1

        def f(w):
            # converts the dense np weight vector into a sparse dict
            qmap = self.nparray2dict(w)
            logging.info('[%d/%d/%d] nonzero weights: %d', self.step, self.iteration, self.funcall, len(qmap))
            # samples from the proxy
            sample = self.sample(qmap)
            # estimate KL(q||p) and its derivatives wrt lambda 
            obj, jac, ne, dne = self.estimates(sample, qmap, self.pmap_)
            logging.info('[%d/%d/%d] KL=%s ne=%f', self.step, self.iteration, self.funcall, obj, ne)
            # regularisation
            if self.regulariser_ is not KLOptimiser.NoRegularisation:
                regulariser = self.regulariser_(w)  # LA.norm(w, self.regulariser_)
                obj = obj + self.regulariser_weight_ * regulariser
                jac = jac + 2 * self.regulariser_weight_ * w
                logging.info('[%d/%d/%d] Regularised KL=%s', self.step, self.iteration, self.funcall, obj)
            self.funcall += 1
            return obj, jac

        def f2(w):
            # converts the dense np weight vector into a sparse dict
            qmap = self.nparray2dict(w)
            logging.info('[%d/%d/%d] nonzero weights: %d', self.step, self.iteration, self.funcall, len(qmap))
            # samples from the proxy
            sample = self.sample(qmap)
            # estimate KL(q||p) and its derivatives wrt lambda
            kl, dkl, ne, dne = self.estimates(sample, qmap, self.pmap_)
            logging.info('[%d/%d/%d] KL=%s ne=%f', self.step, self.iteration, self.funcall, kl, ne)
            obj = -ne
            jac = -dne
            self.funcall += 1
            return obj, jac

        def f3(w):
            # converts the dense np weight vector into a sparse dict
            qmap = self.nparray2dict(w)
            logging.info('[%d/%d/%d] nonzero weights: %d', self.step, self.iteration, self.funcall, len(qmap))
            # samples from the proxy
            sample = self.sample(qmap)
            # estimate KL(q||p) and its derivatives wrt lambda
            kl, dkl, ne, dne = self.estimates(sample, qmap, self.pmap_)
            logging.info('[%d/%d/%d] KL=%s ne=%f', self.step, self.iteration, self.funcall, kl, ne)
            obj = kl
            jac = dkl

            if self.ne_weight_ != 0.0:
                obj -= self.ne_weight_ * ne
                jac -= self.ne_weight_ * dne

            if self.regulariser_ > 0 and self.regulariser_weight_ != 0.0:
                obj += self.regulariser_weight_ * LA.norm(w, self.regulariser_)
                jac += 2 * self.regulariser_weight_ * w

            logging.info('[%d/%d/%d] Regularised KL=%s', self.step, self.iteration, self.funcall, obj)
            self.funcall += 1
            return obj, jac
            
        def callback(w):
            logging.info('New lambdas: %s', w)
            self.iteration += 1

        def gcallback(w, fvalue, a):
            logging.info('New local minimum: %s', fvalue)
            self.step += 1
        
        logging.info('Optimising KL(q||p)')
        if False:
            result = minimize(f,
                    self.dict2nparray(self.qmap0_),  # initial weights
                    method='L-BFGS-B', 
                    jac=True, 
                    callback=callback, 
                    options={'maxiter': 10, 'maxfun': 20, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False})

        minimiser_args = {
                'method': 'L-BFGS-B',
                'jac': True,
                'callback': callback,
                'options': {'maxiter': 5, 'maxfun': 10, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
                }
        result = basinhopping(f3,
                self.dict2nparray(self.qmap0_),  # initial weights
                minimizer_kwargs=minimiser_args,
                niter=3,
                niter_success=2,  # TODO: relax the test (not strictly the same, but close enough for enough iterations)
                callback=gcallback
                )

        qmap = self.nparray2dict(result.x)
        logging.info('Final KL: %s', result.fun)
        logging.info('Nonzero weights: %d', len(qmap))
        #logging.info('Final lambda: %s', fpairs2str(qmap.iteritems()))


        return qmap 
