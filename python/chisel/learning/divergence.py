import os
import logging
import shlex
import chisel.mteval as mteval
import subprocess as sp
import traceback
import sys
import numpy as np
from numpy import linalg as LA

from multiprocessing import Pool
from time import time, strftime
from scipy.optimize import minimize

from ConfigParser import RawConfigParser
from functools import partial
from chisel.util.wmap import WMap, JointWMap
from chisel.learning import risk, divergence
from chisel.util import scaled_fmap, npvec2str
from chisel.util.iotools import SegmentMetaData, list_numbered_files
from chisel.util.config import configure, section_literal_eval


def wrapped_divergence(job, iteration, q_wmap, p_wmap, sample_headers, save_to=None):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    seg, samples_file = job
    try:
        result = divergence(seg=seg,
                samples_file=samples_file,
                q_wmap=q_wmap,
                p_wmap=p_wmap,
                sample_headers=sample_headers,
                save_to=save_to)
        return result
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


class KLDriver(object):

    SAMPLING_HEADERS = {'derivation': 'd', 'vector': 'v', 'count': 'n', 'log_ur': 'log_ur', 'importance': 'importance'}

    def __init__(self, 
            args, config, parent_workspace, workspace, 
            wmap, devset, proxy_scaling, target_scaling, 
            parent_iteration):

        # 1) sanity checks
        # 2) lead metric
        # 3) load model 
        # 4) create workspace
        # 5) load devset
        # 6) prepare devset in workspace
        # 7) base config file
        # 8) iterations

        self.args = args
        self.config = config
        self.parent_workspace = parent_workspace
        self.wmap = wmap
        self.devset = devset 
        self.alpha_q, self.alpha_p = proxy_scaling, target_scaling
        self.parent_iteration = parent_iteration
        self.workspace = KLDriver._MAKE_WORKSPACE_(workspace, 'SGD')
        self.base_config = KLDriver._BASE_CONFIG_(config, self.workspace, self.wmap.proxy, self.wmap.target)

        self.iteration = 0

    def path_to_config(self, iteration=None):
        if iteration is None:  # return the current config
            return '{0}/config{1}.ini'.format(self.workspace, self.iteration)
        elif iteration < 0:  # return the base config
            return '{0}/base_config.ini'.format(self.workspace)
        else:  # return the config requested
            return '{0}/config{1}.ini'.format(self.workspace, iteration)
    
    def current_config(self):
        return self.path_to_config()

    def path_to_run(self):
        return '{0}/run{1}'.format(self.workspace, self.iteration)

    def path_to_samples(self):
        return '{0}/samples'.format(self.path_to_run())
    
    def path_to_estimates(self):
        return '{0}/estimates'.format(self.path_to_run())
    
    def path_to_loss(self):
        return '{0}/loss'.format(self.path_to_run())
    
    def path_to_risk(self):
        return '{0}/risk'.format(self.path_to_run())
    
    def path_to_kl(self):
        return '{0}/KL'.format(self.path_to_run())
    
    def path_to_kl_run(self):
        return '{0}/run{1}'.format(self.path_to_kl(), self.kl_iteration)

    def update_config_file(self, proxy_scaling=1.0, target_scaling=1.0):
        config = RawConfigParser()
        config.optionxform = str

        try:
            config.read(self.path_to_config(self.iteration - 1))
        except IOError as e:
            logging.error('[%d/%d] perhaps the previous iteration did not complete successfully', self.parent_iteration, self.iteration)
            raise e

        config.set('chisel:model', 'proxy_scaling', proxy_scaling)
        config.set('chisel:model', 'target_scaling', target_scaling)

        [config.set('proxy', f, v) for f, v in self.wmap.proxy.iteritems()]
        [config.set('target', f, v) for f, v in self.wmap.target.iteritems()]
    
        with open('{0}/config{1}.ini'.format(self.workspace, self.iteration), 'wb') as fo:
            config.write(fo)
    
    def make_sampling_options(self):
        options = {'config': self.path_to_config(),
                'workspace': self.path_to_run()}
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s' % options
        cmd_args = shlex.split(cmd_str)
        return cmd_args

    def path_to_dev_src(self):
        return '{0}/dev.input'.format(self.parent_workspace)

    def path_to_dev_refs(self):
        return '{0}/dev.refs'.format(self.parent_workspace)

    def path_to_log(self, source, err=False):
        return '{0}/log/{1}.{2}.std{3}'.format(self.workspace, source, self.iteration, 'err' if err else 'out')
    
    def check_samples(self, iteration):
        return True

    def sample(self):
        self.update_config_file(proxy_scaling=self.alpha_q, target_scaling=self.alpha_p)
        t0 = time()
        with open(self.path_to_dev_src(), 'rb') as fi:
            with open(self.path_to_log('sampling'), 'wb') as fo:
                with open(self.path_to_log('sampling', err=True), 'wb') as fe:
                    cmd_args = self.make_sampling_options()
                    proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                    proc.wait()
        dt = time() - t0
        logging.info('[%d/%d] sampling took %f seconds', self.parent_iteration, self.iteration, dt)
        if not self.check_samples(self.iteration):
            raise Exception('chisel.sampler appears to have failed at iteration %d', self.iteration)
        return dt
    
    def KL(self):
        t0 = time()
        # read list of input files
        samples_dir = self.path_to_samples()
        if not os.path.isdir(samples_dir):
            raise Exception('[%d/%d] could not find samples' % (self.parent_iteration, self.iteration))
        #logging.info('[%d] reading samples from %s', self.iteration, samples_dir)
        input_files = list_numbered_files(samples_dir)

        # make jobs
        jobs = [(self.devset[fid], input_file) for fid, input_file in input_files]
        #logging.info('[%d] %d sampling jobs', self.iteration, len(jobs))

        # run jobs in parallel
        pool = Pool(self.args.jobs)
        results = pool.map(partial(wrapped_divergence,
                                   iteration=self.iteration,
                                   q_wmap=self.wmap.proxy,
                                   p_wmap=self.wmap.target,
                                   sample_headers=KLDriver.SAMPLING_HEADERS),
                           jobs)
        # gather risks into an array
        divergences = np.array([result.KL for result in results], float)
        derivatives = np.array([result.dKL for result in results], float)
        dt = time() - t0
        logging.info('[%d/%d] assessing divergence took %f seconds', self.parent_iteration, self.iteration, dt)
        # gather KL
        return divergences, derivatives

    def optimise(self):

        self.iteration = -1

        self.history = []
        self.selected = []
        
        def f(w):

            self.iteration += 1

            # the joint update is no longer supported: self.wmap.update(w)
            self.history.append(w)
            self.wmap.proxy.update(w)
            logging.info('[%d/%d] lambda=%s', self.parent_iteration, self.iteration, npvec2str(w))

            #logging.info('[%d/%d] sampling ...', self.parent_iteration, self.iteration)
            self.sample()
           
            #logging.info('[%d/%d] KL ...', self.parent_iteration, self.iteration)
            divergences, derivatives = self.KL()

            kl = divergences.mean(0)
            dkl = derivatives.mean(0)
            # r_weight
            r_weight = 0
            # regularised
            regulariser = LA.norm(w, 2)

            r_obj = kl + r_weight * regulariser
            r_jac = dkl + 2 * r_weight * w
            
            logging.info('[%d/%d] KL=%s regularised=%f', self.parent_iteration, self.iteration, kl, r_obj)

            return r_obj, r_jac
            

        def callback(w):
            self.selected.append(w)
            logging.info('[%d/%d] new lambda: %s', self.parent_iteration, self.iteration, npvec2str(w))

        self.result_ = minimize(f, 
                self.wmap.proxy.asarray, 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxfun': 10, 'ftol': 1e-6, 'gtol': 1e-3, 'disp': False}) # TODO find a way to stop the search earlier 'maxfun'

        logging.info('[%d/%d] final KL=%s lambda=%s', self.parent_iteration, self.iteration, self.result_.fun, npvec2str(self.result_.x))
        
        return self.result_.x, self.result_.fun

    def optimise_scaling(self):

        self.iteration = -1

        self.history = []
        self.selected = []
       
        target = self.wmap.target.asdict()
        self.base_ = np.array([target[f] for f in self.wmap.proxy.features])
        
        def f(w):

            self.iteration += 1

            # the joint update is no longer supported: self.wmap.update(w)
            self.history.append(w)
            self.wmap.proxy.update(w * self.base_)
            logging.info('[%d/%d] alpha=%s', self.parent_iteration, self.iteration, w)

            logging.info('[%d/%d] sampling ...', self.parent_iteration, self.iteration)
            self.sample()
           
            logging.info('[%d/%d] KL ...', self.parent_iteration, self.iteration)
            divergences, _ = self.KL()

            kl = divergences.mean(0)
            # r_weight
            r_weight = 0
            # regularised
            regulariser = LA.norm(w, 2)

            r_obj = kl + r_weight * regulariser
            
            logging.info('[%d/%d] kl=%s regularised=%f', self.parent_iteration, self.iteration, kl, r_obj)

            return r_obj
            

        def callback(w):
            self.selected.append(w)
            logging.info('[%d/%d] new alpha: %s', self.parent_iteration, self.iteration, str(w))

        logging.info('Minimising')
        self.result_ = minimize(f, 
                np.array([1.0]), 
                method='Powell', 
                callback=callback, 
                #bounds=[(0,1)],
                options={'maxfev': 10}) # TODO find a way to stop the search earlier
        print self.result_

        return self.result_.x * self.base_

    @staticmethod
    def _START_MODEL_(config):
        """
        Requires: config, config_learning
        Produces: wmap
        """
        # parameters of the instrumental distribution
        proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')))
        # parameters of the target distribution
        target_weights = scaled_fmap(section_literal_eval(config.items('target')))

        return JointWMap(WMap(sorted(proxy_weights.iteritems(), key=lambda (k, v): k)),
                WMap(sorted(target_weights.iteritems(), key=lambda (k, v): k)))

    @staticmethod
    def _MAKE_WORKSPACE_(workspace, algorithm):
        """
        Produces: workspace
        """
        path = '{0}/KL-{1}'.format(workspace, 
                algorithm) 
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists('{0}/log'.format(path)):
            os.makedirs('{0}/log'.format(path))
        return path

    @staticmethod
    def _BASE_CONFIG_(config, workspace, proxy_wmap, target_wmap):
        if config.has_section('proxy'):
            config.remove_section('proxy')
        config.add_section('proxy')
        [config.set('proxy', f, v) for f, v in proxy_wmap.iteritems()]
        
        if config.has_section('target'):
            config.remove_section('target')
        config.add_section('target')
        [config.set('target', f, v) for f, v in target_wmap.iteritems()]
        
        with open('{0}/base_config.ini'.format(workspace), 'wb') as fo:
            config.write(fo)

        return '{0}/base_config.ini'.format(workspace)

