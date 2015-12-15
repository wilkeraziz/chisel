import io
import os
from os.path import splitext
import logging
import shlex
import chisel.mteval as mteval
import subprocess as sp
import traceback
import sys
import numpy as np
from numpy import linalg as LA
from collections import namedtuple
from multiprocessing import Pool
from time import time, strftime
from scipy.optimize import minimize
from ConfigParser import RawConfigParser
from functools import partial

from chisel.learning import risk, divergence
from chisel.decoder.estimates import EmpiricalDistribution
from chisel.util.wmap import WMap, JointWMap
from chisel.util import scaled_fmap, npvec2str
from chisel.util.iotools import SegmentMetaData, list_numbered_files, smart_ropen, smart_wopen
from chisel.util.iotools import sampled_derivations_from_file
from chisel.util.config import configure, section_literal_eval
from chisel.learning.divergence import KLDriver
import chisel.learning.newestimates as optalg
from chisel.mteval.fast_bleu import TrainingBLEU


MTEvalHyp = namedtuple('MTEvalHyp', ['projection', 'leaves'])

class RunCount(object):

    def __init__(self, run=0, risk=0, kl=0):
        self.run = run
        self.risk = risk
        self.kl = kl

    def __str__(self):
        return '{0}/{1}/{2}'.format(self.run, self.risk, self.kl)

    @property
    def iteration(self):
        return self.run

    def next_risk(self):
        self.risk += 1

    def next_kl(self):
        self.kl += 1

    def next_run(self):
        self.run += 1
        self.risk = 0
        self.kl = 0

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

class Driver(object):

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
        self.workspace = Driver._MAKE_WORKSPACE_(self.args.workspace, 'SGD', self.args.metric, self.args.alias)

        if args.skip == 0:
            self.wmap = Driver._START_MODEL_(self.config, default=self.args.default)
            Driver._BASE_CONFIG_(self.config, self.workspace, self.wmap.proxy, self.wmap.target)
        else:
            logging.info('Loading model %s', '{0}/config{1}.ini'.format(self.workspace, args.skip))
            self.wmap = self.configure('{0}/config{1}.ini'.format(self.workspace, args.skip))

        self.devset = Driver._PREPARE_DEVSET_(self.workspace, self.args.dev, config, 'dev')
        self.devtest = None
        if args.devtest:  # load devtest set if given
            self.devtest = Driver._PREPARE_DEVSET_(self.workspace, self.args.devtest, config, 'devtest', grammar_dir=args.devtest_grammar)

        Driver._LOAD_METRIC_(self.config, self.args.metric)
        self.tune()

    def update_config_file(self, number, proxy_scaling=None, target_scaling=None, proxy=None, target=None):
        config = RawConfigParser()
        config.optionxform = str
        
        try:
            config.read('{0}/config{1}.ini'.format(self.workspace, number))
        except IOError as e:
            logging.error('Perhaps iteration %d did not complete successfully', number)
            raise e

        if proxy_scaling is not None:
            config.set('chisel:model', 'proxy_scaling', proxy_scaling)
        if target_scaling is not None:
            config.set('chisel:model', 'target_scaling', target_scaling)

        if proxy is None:
            [config.set('proxy', f, v) for f, v in self.wmap.proxy.iteritems()]
        else:
            [config.set('proxy', f, v) for f, v in proxy.iteritems()]
        if target is None:
            [config.set('target', f, v) for f, v in self.wmap.target.iteritems()]
        else:
            [config.set('target', f, v) for f, v in target.iteritems()]
    
        with smart_wopen('{0}/config{1}.ini'.format(self.workspace, number + 1)) as fo:
            config.write(fo)
    
    def path_to_log(self, source, iteration, err=False):
        return '{0}/log/{1}.{2}.std{3}.gz'.format(self.workspace, source, iteration, 'err' if err else 'out')
    
    def sample(self, iteration):
        options = {'config': '{0}/config{1}.ini'.format(self.workspace, iteration - 1), 
                'workspace': '{0}/run{1}'.format(self.workspace, iteration)}
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s' % options
        cmd_args = shlex.split(cmd_str)

        t0 = time()
        logging.info('[%d] Sampling...', iteration)
        with smart_ropen('{0}/dev.input'.format(self.workspace)) as fi:
            with smart_wopen(self.path_to_log('sampling', iteration)) as fo:
                with smart_wopen(self.path_to_log('sampling', iteration, err=True)) as fe:
                    proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                    proc.wait()
        dt = time() - t0
        logging.info('[%d] sampling took %f seconds', iteration, dt)
        #if not self.check_samples(self.iteration):
        #    raise Exception('chisel.sampler appears to have failed at iteration %d', run.iteration)
        return '{0}/run{1}/samples'.format(self.workspace, iteration)
    
    
    def mteval(self, iteration):
        # sample
        options = {'config': '{0}/config{1}.ini'.format(self.workspace, iteration), 
                'workspace': '{0}/run{1}/devtest'.format(self.workspace, iteration)}
        mkdir(options['workspace'])
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s' % options
        if self.args.devtest_grammar:
            cmd_str = '{0} --grammar {1}'.format(cmd_str, self.args.devtest_grammar)
        cmd_args = shlex.split(cmd_str)
        t0 = time()
        logging.info('[%d] Sampling (devtest)...', iteration)
        with smart_ropen('{0}/devtest.input'.format(self.workspace)) as fi:
            with smart_wopen(self.path_to_log('sampling-devtest', iteration)) as fo:
                with smart_wopen(self.path_to_log('sampling-devtest', iteration, err=True)) as fe:
                    proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                    proc.wait()
        dt = time() - t0
        logging.info('[%d] sampling took %f seconds', iteration, dt)
        # decision rule
        cmd_str = 'python -m chisel.fast_consensus %(config)s %(workspace)s ' % options
        logging.info('[%d] deciding (devtest): %s', iteration, cmd_str)
        cmd_args = shlex.split(cmd_str)
        t0 = time()
        with smart_wopen(self.path_to_log('decision-devtest', iteration)) as fo:
            with smart_wopen(self.path_to_log('decision-devtest', iteration, err=True)) as fe:
                proc = sp.Popen(cmd_args, stdin=None, stdout=fo, stderr=fe)
                proc.wait()
        dt = time() - t0
        logging.info('[%d] deciding took %f seconds', iteration, dt)
        # mt eval
        cmd_str = '{0} -r {1}'.format(self.args.scoring_tool, '{0}/devtest.refs'.format(self.workspace))
        cmd_args = shlex.split(cmd_str)
        trans_path = '{0}/output/consensus-bleu'.format(options['workspace'])
        with smart_ropen(trans_path) as fin:
            bleu_out = '{0}.bleu.stdout'.format(splitext(trans_path)[0])
            bleu_err = '{0}.bleu.stderr'.format(splitext(trans_path)[0])
            with smart_wopen(bleu_out) as fout:
                with smart_wopen(bleu_err) as ferr:
                    logging.info(cmd_args)
                    proc = sp.Popen(cmd_args, stdin=fin, stdout=fout, stderr=ferr)
                    proc.wait()
                    try:
                        with smart_ropen(bleu_out) as fi:
                            line = next(fi)
                            return float(line.strip())
                    except:
                        logging.error('Problem reading %s', bleu_out)
        return None

    
    def configure(self, path):
        """
        :param path: path to a configuration file containing weights
        :returns: WMap (proxy and target weights)
        """
        
        # load a given configuration file (with weights)
        config = RawConfigParser()
        config.optionxform = str
        try:
            config.read(path)
        except IOError as e:
            logging.error('Perhaps iteration %d did not complete successfully', number)
            raise e
    
        # parameters of the instrumental distribution
        proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')))

        # parameters of the target distribution
        target_weights = scaled_fmap(section_literal_eval(config.items('target')))

        return JointWMap(WMap(sorted(proxy_weights.iteritems(), key=lambda (k, v): k)),
                WMap(sorted(target_weights.iteritems(), key=lambda (k, v): k)))
    
    def tune(self):

        metric = self.args.metric
        devset = self.devset

        run = RunCount(self.args.skip)
            
        if self.args.skip == 0:
            eval_score = self.mteval(run.iteration)
            logging.info('[%d] Devtest eval: %s', run.iteration, eval_score)

        for loop in range(self.args.skip, self.args.maxiter):  # TODO: implement skip, eval devtest
            run.next_run()
            path_to_run = '{0}/run{1}'.format(self.workspace, run.iteration)
            # sample
            self.sample(run.iteration)  # sample in parallel
            samples_dir = '{0}/samples'.format(path_to_run)
            if not os.path.isdir(samples_dir):
                raise Exception('[%d] could not find samples' % run.iteration)
            logging.info('[%d] reading samples from %s', run.iteration, samples_dir)
            input_files = list_numbered_files(samples_dir)
            S = []
            for fid, input_file in input_files:
                logging.debug(' reading %s from %s', fid, input_file)
                derivations, _qmap, _pmap = sampled_derivations_from_file(input_file)
                S.append(derivations)
            # compute loss
            save_losses = False
            L = []
            loss_dir = '{0}/loss'.format(path_to_run)
            mkdir(loss_dir)

            logging.info('Running fast_bleu')
            for i, derivations in enumerate(S):
                seg = devset[i]
                projections = frozenset(d.tree.projection for d in derivations)
                scorer = TrainingBLEU(seg.refs)
                lmap = {y: scorer.loss(y.split()) for y in projections}
                L.append(lmap)
                if save_losses:
                    with smart_wopen('{0}/{1}.gz'.format(loss_dir, seg.id)) as fo:
                        for d in derivations:
                            fo.write('{0} {1}\n'.format(lmap[d.tree.projection]))
            logging.info('fast_bleu finished')

            #############
            if False:
                logging.info('[%s] Computing loss against references', run.iteration)
                for i, derivations in enumerate(S):  # TODO: compute loss in parallel
                    seg = devset[i]
                    logging.debug(' assessing %s', seg.id)
                    projections = frozenset(d.tree.projection for d in derivations)
                    hyps = [MTEvalHyp(y, tuple(y.split())) for y in projections]
                    #hyps = [d.tree for d in derivations]
                    mteval.prepare_training(seg.src, seg.refs, hyps)
                    lmap = {hyp.projection: mteval.training_loss(c=h, metric=metric) for h, hyp in enumerate(hyps)}
                    #losses = np.array([mteval.training_loss(c=h, metric=metric) for h in range(len(hyps))], float)
                    L.append(lmap)
                    # saves losses
                    if save_losses:
                        with smart_wopen('{0}/{1}.gz'.format(loss_dir, seg.id)) as fo:
                            for d in derivations:
                                fo.write('{0} {1}\n'.format(lmap[d.tree.projection]))
                logging.info('slow_bleu finished')
            #################

            # tuning iteration
            if self.args.order == 'pq':
                target_weights = self.optimise_target(run, devset, S, L)
                self.wmap.target.update(target_weights)
                proxy_weights = self.optimise_proxy(run, devset, S)
                self.wmap.proxy.update(proxy_weights)
            else:
                proxy_weights = self.optimise_proxy(run, devset, S)
                self.wmap.proxy.update(proxy_weights)
                target_weights = self.optimise_target(run, devset, S, L)
                self.wmap.target.update(target_weights)
            # update the config that precedes this run
            self.update_config_file(run.iteration - 1)
            eval_score = self.mteval(run.iteration)
            logging.info('[%d] Devtest eval: %s', run.iteration, eval_score)

    def optimise_target(self, run, devset, S, L):
                                                
        def f(theta):

            run.next_risk()
            
            self.wmap.target.update(theta)
            logging.info('[%s] theta=%s', run, npvec2str(theta))

            emp_risk = []
            emp_dR = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                lmap = L[i]
                #empdist = EmpiricalDistribution(derivations,
                #                                q_wmap=self.wmap.proxy,
                #                                p_wmap=self.wmap.target,
                #                                empirical_q=True,  # crucial: the proposal is fixed, thus we can rely on empirical estimates
                #                                get_yield=lambda d: d.tree.projection)
                support, posterior, dP = optalg.minrisk(derivations, q_wmap=self.wmap.proxy, p_wmap=self.wmap.target, empirical_q=True, get_yield=lambda d: d.tree.projection)

                losses = np.array([lmap[Dy.projection] for Dy in support], float)  # l(y)
                #posterior = empdist.copy_posterior()  # p(y)
                #dP = empdist.copy_dpdt() #  dp(y)/dt
                dR = losses.dot(dP) 
                risk = losses.dot(posterior.transpose())
                emp_risk.append(risk)
                emp_dR.append(dR)

            obj, jac = np.mean(emp_risk, 0), np.mean(emp_dR, 0)

            # r_weight
            r_weight = self.args.riskreg
            if r_weight != 0.0:  # regularised
                regulariser = LA.norm(theta, 2)
                r_obj = obj + r_weight * regulariser
                r_jac = jac + 2 * r_weight * theta
                logging.info('[%s] RISK=%f regularised=%f', run, obj, r_obj)
                return r_obj, r_jac
            else:
                logging.info('[%s] RISK=%f', run, obj)
                return obj, jac

        def callback(theta):
            logging.info('[%s] new theta: %s', run, npvec2str(theta))

        logging.info('[%s] Minimising risk', run)
        result = minimize(f, 
                self.wmap.target.asarray, 
                #method='BFGS', 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxiter': self.args.rsgd[0],
                    'ftol': 1e-4,
                    'gtol': 1e-4,
                    'maxfun': self.args.rsgd[1],
                    'disp': False})
        logging.info('[%s] Done!', run)
        return result.x
    
    def optimise_proxy(self, run, devset, S):
        
        if self.args.method == 'minkl':
            logging.info('[%s] Minimising KL divergence', run)
            get_obj_and_jac = optalg.minkl
            polarity = 1
        elif self.args.method == 'maxelb':
            logging.info('[%s] Maximising ELB', run)
            get_obj_and_jac = optalg.maxelb
            polarity = -1
        else:
            raise NotImplementedError('Unsupported optimisation method: %s', self.args.method)
        #elif self.args == 'minvar': 
        #    pass

                                                
        def f(theta):

            run.next_kl()
            self.wmap.proxy.update(theta)
            logging.info('[%s] lambda=%s', run, npvec2str(theta))

            OBJ = []
            JAC = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                #empdist = EmpiricalDistribution(derivations,
                #                                q_wmap=self.wmap.proxy,
                #                                p_wmap=self.wmap.target,
                #                                empirical_q=False,  # crucial: the proposal is changing, thus we cannot rely on empirical estimates
                #                                get_yield=lambda d: d.tree.projection)

                local_obj, local_jac = get_obj_and_jac(derivations, q_wmap=self.wmap.proxy, p_wmap=self.wmap.target, empirical_q=False)
                
                #elb, delb = empdist.elb()
                #kl, dkl = empdist.kl()
                # TODO: min KL  or min - ELB (should be equivalent, shouldn't it?)
                OBJ.append(local_obj)
                JAC.append(local_jac)
                #hq, dhq = empdist.Hq()
                #H.append(hq)
                #dH.append(dhq)

            obj, jac = np.mean(OBJ, 0), np.mean(JAC, 0)
            if polarity != 1:
                obj *= polarity
                jac *= polarity
            
            #if self.args.temperature != 0.0:
            #    obj -= T * np.mean(H, 0)
            #    jac -= T * np.mean(dH, 0)

            r_weight = self.args.klreg
            if r_weight != 0.0:  # regularised
                regulariser = LA.norm(theta, 2)
                r_obj = obj + r_weight * regulariser
                r_jac = jac + 2 * r_weight * theta
                logging.info('[%s] O=%f regularised=%f', run, obj, r_obj)
                return r_obj, r_jac
            else:
                logging.info('[%s] O=%f', run, obj)
                return obj, jac
            

        def callback(theta):
            logging.info('[%s] new lambda: %s', run, npvec2str(theta))

        result = minimize(f, 
                self.wmap.proxy.asarray, 
                #method='BFGS', 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxiter': self.args.klsgd[0],
                    'ftol': 1e-4,
                    'gtol': 1e-4,
                    'maxfun': self.args.klsgd[1],
                    'disp': False})
        logging.info('[%s] Done!', run)
        return result.x


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
        
        with smart_wopen('{0}/config0.ini'.format(workspace)) as fo:
            config.write(fo)

        return '{0}/config0.ini'.format(workspace)

    @staticmethod
    def _PREPARE_DEVSET_(workspace, path, config, stem='dev', input_format='cdec', grammar_dir=None):
        # load dev set and separate input and references
        logging.info('Reading %s set: %s', stem, path)
        if grammar_dir is None:
            if config.has_section('chisel:sampler'):
                sampler_map = section_literal_eval(config.items('chisel:sampler'))
                grammar_dir = sampler_map.get('grammars', None)
        with smart_ropen(path) as f:
            devset = [SegmentMetaData.parse(line.strip(),
                                              input_format,
                                              grammar_dir=grammar_dir)
                        for sid, line in enumerate(f)]
        logging.info('%d %s instances', len(devset), stem)

        # dump source and references
        with smart_wopen('{0}/{1}.input'.format(workspace, stem)) as fi:
            with smart_wopen('{0}/{1}.refs'.format(workspace, stem)) as fr:
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

