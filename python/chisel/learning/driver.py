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
from chisel.learning import risk, divergence
from chisel.util.wmap import WMap, JointWMap
from chisel.util import scaled_fmap, npvec2str
from chisel.util.iotools import SegmentMetaData, list_numbered_files
from chisel.util.config import configure, section_literal_eval
from chisel.learning.divergence import KLDriver

def wrapped_risk(job, iteration, q_wmap, p_wmap, metric, sample_headers, consensus=False, save_to=None):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    seg, samples_file = job
    try:
        t0 = time()
        result = risk(seg=seg,
                samples_file=samples_file,
                q_wmap=q_wmap,
                p_wmap=p_wmap,
                metric=metric,
                sample_headers=sample_headers,
                consensus=consensus,
                save_to=save_to)
        dt = time() - t0
        #logging.debug('[%d] (%d) computing risk took %f seconds: risk=%f', iteration, seg.id, dt, result.risk)
        return result
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


class Driver(object):

    SAMPLING_HEADERS = {'derivation': 'd', 'vector': 'v', 'count': 'n', 'log_ur': 'log_ur', 'importance': 'importance'}

    def __init__(self, args, config):

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
    
        self.alpha_p = 1.0 # 0.1
        self.alpha_q = 1.0 # 0.1
        self.wmap = Driver._START_MODEL_(self.config, default=self.args.default)
        self.workspace = Driver._MAKE_WORKSPACE_(self.args.workspace, 'SGD', self.args.metric, self.args.alias)
        self.base_config = Driver._BASE_CONFIG_(self.config, self.workspace, self.wmap.proxy, self.wmap.target)
        self.devset = Driver._PREPARE_DEVSET_(self.workspace, self.args.dev, config, 'dev')
        Driver._LOAD_METRIC_(self.config, self.args.metric)

        self.iteration = 0
        self.optimise()

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
    
    def update_config_file(self, proxy_scaling=1.0, target_scaling=1.0):
        config = RawConfigParser()
        config.optionxform = str

        try:
            config.read(self.path_to_config(self.iteration - 1))
        except IOError as e:
            logging.error('[%d] perhaps the previous iteration did not complete successfully', self.iteration)
            raise e

        config.set('chisel:model', 'proxy_scaling', proxy_scaling)
        config.set('chisel:model', 'target_scaling', target_scaling)

        [config.set('proxy', f, v) for f, v in self.wmap.proxy.iteritems()]
        [config.set('target', f, v) for f, v in self.wmap.target.iteritems()]
        #theta = self.theta.asdict()
        #[config.set('proxy', f, theta[f]) for f in self.wmap.proxy.features]
        #[config.set('target', f, v) for f, v in theta.iteritems()]
    
        with open('{0}/config{1}.ini'.format(self.workspace, self.iteration), 'wb') as fo:
            config.write(fo)
    
    def make_sampling_options(self):
        options = {'config': self.path_to_config(),
                'workspace': self.path_to_run()}
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s' % options
        cmd_args = shlex.split(cmd_str)
        return cmd_args

    def make_decision_options(self, rules='--mbr'):
        options = {'config': self.path_to_config(),
                'workspace': self.path_to_run(),
                'rules': rules} 
        cmd_str = 'python -m chisel.decision %(config)s %(workspace)s %(rules)s' % options
        logging.info('[%d] deciding: %s', self.iteration, cmd_str)
        cmd_args = shlex.split(cmd_str)
        return cmd_args
    
    def path_to_dev_src(self):
        return '{0}/dev.input'.format(self.workspace)

    def path_to_dev_refs(self):
        return '{0}/dev.refs'.format(self.workspace)

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
        logging.info('[%d] sampling took %f seconds', self.iteration, dt)
        if not self.check_samples(self.iteration):
            raise Exception('chisel.sampler appears to have failed at iteration %d', self.iteration)
        return dt
    
    def check_decisions(self, iteration):
        return True

    def decide(self):
        t0 = time()
        with open(self.path_to_log('decision'), 'wb') as fo:
            with open(self.path_to_log('decision', err=True), 'wb') as fe:
                cmd_args = self.make_decision_options()
                proc = sp.Popen(cmd_args, stdin=None, stdout=fo, stderr=fe)
                proc.wait()
        dt = time() - t0
        logging.info('[%d] deciding took %f seconds', self.iteration, dt)
        if not self.check_decisions(self.iteration):
            raise Exception('chisel.decision appears to have failed at iteration %d', self.iteration)
        return dt

    def risk(self):
        t0 = time()
        # read list of input files
        samples_dir = self.path_to_samples()
        if not os.path.isdir(samples_dir):
            raise Exception('[%d] could not find samples' % self.iteration)
        #logging.info('[%d] reading samples from %s', self.iteration, samples_dir)
        input_files = list_numbered_files(samples_dir)

        # make jobs
        jobs = [(self.devset[fid], input_file) for fid, input_file in input_files]
        #logging.info('[%d] %d sampling jobs', self.iteration, len(jobs))

        if not os.path.exists(self.path_to_estimates()):
            os.makedirs(self.path_to_estimates())
        if not os.path.exists(self.path_to_loss()):
            os.makedirs(self.path_to_loss())
        if not os.path.exists(self.path_to_risk()):
            os.makedirs(self.path_to_risk())

        # run jobs in parallel
        pool = Pool(self.args.jobs)
        results = pool.map(partial(wrapped_risk,
                                   iteration=self.iteration,
                                   q_wmap=self.wmap.proxy,
                                   p_wmap=self.wmap.target,
                                   metric=self.args.metric,
                                   sample_headers=Driver.SAMPLING_HEADERS,
                                   save_to=(self.path_to_estimates(), self.path_to_loss(), self.path_to_risk())),
                           jobs)
        # gather risks into an array
        risks = np.array([result.R for result in results])
        # gather jacobias into an array
        #jacs = np.array([self.wmap.concatenate(proxy=result.dl, target=result.dt) for result in results])
        jacs = np.array([result.dR for result in results])
        dt = time() - t0
        logging.info('[%d] assessing risk took %f seconds', self.iteration, dt)
        # gather KL
        kls = np.array([result.KL for result in results])
        return risks, jacs, kls

    def optimise(self):

        self.iteration = -1
        self.selected_thetas = [self.wmap.target.copy_array()]
        self.selected_lambdas = [self.wmap.proxy.copy_array()]


        self.kl_min = 0
        self.kl_current = 0
        self.kl_coeff = 2.0
        self.kl_lastit = 0
        self.last_callback = 0
        
        def f(theta):

            self.iteration += 1
            
            # the joint update is no longer supported: self.wmap.update(w)
            self.wmap.target.update(theta)

            logging.info('[%d] theta=%s', self.iteration, npvec2str(theta))
         
            # KL here?
            # optimises KL if necessary
            if self.kl_current > self.kl_min * self.kl_coeff or self.iteration > self.kl_lastit + 10:
                lambdas, self.kl_min = self.optimise_proxy(target=theta)
                self.kl_lastit = self.iteration
                # register the lambda
                self.wmap.proxy.update(lambdas)
                self.selected_lambdas.append(lambdas)

            #logging.info('[%d] sampling ...', self.iteration)
            self.sample()
           
            # logging.info('[%d] decision rules ...', self.iteration)
            # self.decide()
            # compute loss

            #logging.info('[%d] assessing risk ...', self.iteration)
            risks, jacs, kls = self.risk()

            risk = risks.mean(0) 
            jac = jacs.mean(0)
            kl = kls.mean(0)
            if kl < self.kl_min:
                self.kl_min = kl
            self.kl_current = kl

            # r_weight
            r_weight = 0
            # regularised
            regulariser = LA.norm(theta, 2)
            r_obj = risk + r_weight * regulariser
            r_jac = jac + 2 * r_weight * theta
            
            logging.info('[%d] RISK=%f regularised=%f KL=%f', self.iteration, risk, r_obj, kl)
            #logging.info('[%d] jac=%s', self.iteration, jac)
            
            return r_obj, r_jac

        def callback(theta):
            logging.info('[%d] new theta: %s', self.iteration, npvec2str(theta))
            # register the theta
            self.wmap.target.update(theta)
            self.selected_thetas.append(theta)
            self.last_callback = self.iteration



        logging.info('Minimising risk')
        self.result_ = minimize(f, 
                self.wmap.target.asarray, 
                #method='BFGS', 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxiter': 20,
                    'ftol': 1e-9,
                    'gtol': 1e-9,
                    'maxfun': 100,
                    'disp': False})

        print self.result_
        return self.result_.x


    def optimise_proxy(self, proxy=None, target=None):
        # selects weight vectors
        wmap = self.wmap.copy() 
        if proxy is not None:
            wmap.proxy.update(proxy)
        if target is not None:
            wmap.target.update(target)
        
        # instantiate a KL driver
        driver = KLDriver(self.args, 
                self.config, 
                self.workspace, 
                self.path_to_run(), 
                wmap,
                self.devset, 
                proxy_scaling=self.alpha_q,
                target_scaling=self.alpha_p,
                parent_iteration=self.iteration)
        
        # optimises
        logging.info('Minimising KL(lambda): min=%f current=%f lastit=%d', self.kl_min, self.kl_current, self.kl_lastit)
        lambdas, kl = driver.optimise()
        #self.wmap.proxy.update(lambdas)
        logging.info('going back to minimising RISK(theta)')
        
        return lambdas, kl

    @staticmethod
    def _START_MODEL_(config, default=None, overwrite_q=False):
        """
        Requires: config, config_learning
        Produces: wmap
        """
        # parameters of the instrumental distribution
        proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')))
        if default is not None:
            proxy_weights = {k: default for k, v in proxy_weights.iteritems()}

        # parameters of the target distribution
        target_weights = scaled_fmap(section_literal_eval(config.items('target')))
        if default is not None:
            target_weights = {k: default for k, v in target_weights.iteritems()}

        # overwrite q
        if overwrite_q:
            proxy_weights = {f: target_weights[f] for f, v in proxy_weights.iteritems()}

        if len(frozenset(proxy_weights.iterkeys()) - frozenset(target_weights.iterkeys())) > 0:
            raise ValueError('The features in q(d) should be a subset of the features in p(d)')

        return JointWMap(WMap(sorted(proxy_weights.iteritems(), key=lambda (k, v): k)),
                WMap(sorted(target_weights.iteritems(), key=lambda (k, v): k)))

    @staticmethod
    def _MAKE_WORKSPACE_(workspace, algorithm, metric, alias):
        """
        Requires: args, config_learning
        Produces: workspace
        """
        if alias is None:
            alias = strftime('%Y%m%d-%H%M%S')
        path = '{0}/tuning/{1}-{2}-{3}'.format(workspace, 
                algorithm, 
                metric, 
                alias)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise Exception('Directory already exists: %s', path)
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

    @staticmethod
    def _PREPARE_DEVSET_(workspace, path, config, stem='dev', input_format='cdec'):
        # load dev set and separate input and references
        logging.info('Reading dev set: %s', path)
        grammar_dir = None
        if config.has_section('chisel:sampler'):
            sampler_map = section_literal_eval(config.items('chisel:sampler'))
            grammar_dir = sampler_map.get('grammars', None)
        with open(path, 'r') as f:
            devset = [SegmentMetaData.parse(line.strip(),
                                              input_format,
                                              sid=sid,
                                              grammar_dir=grammar_dir)
                        for sid, line in enumerate(f)]
        logging.info('%d training instances', len(devset))

        # dump source and references
        with open('{0}/{1}.input'.format(workspace, stem), 'wb') as fi:
            with open('{0}/{1}.refs'.format(workspace, stem), 'wb') as fr:
                for seg in devset:
                    print >> fi, seg.to_sgm(dump_refs=False)
                    print >> fr, ' ||| '.join(str(ref) for ref in seg.refs)

        return devset
    
    @staticmethod
    def _LOAD_METRIC_(config, metric):
        # loads mteval modules
        if config.has_section('chisel:metrics'):
            metrics_map = section_literal_eval(config.items('chisel:metrics'))
        else:
            metrics_map = {'bleu': 'chisel.mteval.bleu'}
        mteval.load(metrics_map, frozenset([metric]))

        if not mteval.sanity_check(metric):
            raise Exception("Perhaps you forgot to include the metric '%s' in the configuration file?" % metric)

        # configure mteval metrics
        if config.has_section('chisel:metrics:config'):
            metrics_config = section_literal_eval(config.items('chisel:metrics:config'))
        else:
            metrics_config = {}
        logging.debug('chisel:metrics:config: %s', metrics_config)
        mteval.configure(metrics_config)

