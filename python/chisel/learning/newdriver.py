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
from collections import namedtuple, deque
from multiprocessing import Pool
from time import time, strftime
from scipy.optimize import minimize
from functools import partial

from chisel.learning import risk, divergence
from chisel.decoder.estimates import EmpiricalDistribution
from chisel.util.wmap import WMap, JointWMap
from chisel.util import scaled_fmap, npvec2str
from chisel.util.iotools import SegmentMetaData, list_numbered_files, smart_ropen, smart_wopen
from chisel.util.iotools import sampled_derivations_from_file
from chisel.util.config import configure, Config
from chisel.learning.divergence import KLDriver
import chisel.learning.newestimates as optalg
from chisel.mteval.fast_bleu import TrainingBLEU


MTEvalHyp = namedtuple('MTEvalHyp', ['projection', 'leaves'])

class RunCount(object):

    def __init__(self, run=0, target=0, proxy=0, targetsgd=0, proxysgd=0):
        self.run = run
        self.target = target
        self.proxy = proxy
        self.targetsgd = targetsgd
        self.proxysgd = proxysgd

    def __str__(self):
        return 'I={0}/P={1}:{2}/Q={3}:{4}'.format(self.run, self.target, self.targetsgd, self.proxy, self.proxysgd)

    @property
    def iteration(self):
        return self.run

    def next_target(self):
        self.target += 1
        self.targetsgd = 0

    def next_proxy(self):
        self.proxy += 1
        self.proxysgd = 0
    
    def next_targetsgd(self):
        self.targetsgd += 1

    def next_proxysgd(self):
        self.proxysgd += 1

    def next_run(self):
        self.run += 1
        self.target = 0
        self.proxy = 0
        self.targetsgd = 0
        self.proxysgd = 0

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
        self.sampling_schedule = deque(self.args.samples)
        self.sampling_schedule_iteration = 0
    
        self.alpha_p = 1.0 # 0.1
        self.alpha_q = 1.0 # 0.1
        self.workspace = Driver._MAKE_WORKSPACE_(self.args.workspace, 'SGD', self.args.metric, self.args.alias)

        if args.resume == 0:
            self.wmap = Driver._START_MODEL_(self.config, default=self.args.default)
            Driver._BASE_CONFIG_(self.config, self.workspace, self.wmap.proxy, self.wmap.target)
        else:
            logging.info('Loading model %s', '{1}/config{1}.ini'.format(self.workspace, args.resume - 1))
            self.wmap = self.configure('{0}/config{1}.ini'.format(self.workspace, args.resume - 1))

        self.devset = Driver._PREPARE_DEVSET_(self.workspace, self.args.dev, config, alias=self.args.dev_alias)
        self.devtest = None
        if args.devtest:  # load devtest set if given
            self.devtest = Driver._PREPARE_DEVSET_(self.workspace, self.args.devtest, config, alias=self.args.devtest_alias, grammar_dir=args.devtest_grammar)

        Driver._LOAD_METRIC_(self.config, self.args.metric)
        t0 = time()
        self.tune()
        dt = time() - t0
        logging.info('Optimisation took %s minutes', dt/60)

    def update_config_file(self, before, after, proxy_scaling=None, target_scaling=None, proxy=None, target=None):
        config_path = '{0}/{1}'.format(self.workspace, before)
        if not os.path.exists(config_path):
            raise IOError('Perhaps iteration %s did not complete successfully?' % path)
        
        config = Config(config_path)

        config.add_section('chisel:model')
        if proxy_scaling is not None:
            config.set('chisel:model', 'proxy_scaling', proxy_scaling)
        if target_scaling is not None:
            config.set('chisel:model', 'target_scaling', target_scaling)

        config.add_section('proxy')
        if proxy is None:
            [config.set('proxy', f, v) for f, v in self.wmap.proxy.iteritems()]
        else:
            [config.set('proxy', f, v) for f, v in proxy.iteritems()]

        config.add_section('target')
        if target is None:
            [config.set('target', f, v) for f, v in self.wmap.target.iteritems()]
        else:
            [config.set('target', f, v) for f, v in target.iteritems()]
    
        config_path = '{0}/{1}'.format(self.workspace, after)
        with smart_wopen(config_path) as fo:
            config.write(fo)
        return config_path
    
    def path_to_log(self, source, iteration, err=False):
        return '{0}/log/{1}.{2}.std{3}'.format(self.workspace, source, str(iteration).replace('/', '_'), 'err' if err else 'out')
    
    def configure(self, path):
        """
        :param path: path to a configuration file containing weights
        :returns: WMap (proxy and target weights)
        """
        
        # load a given configuration file (with weights)
        if not os.path.exists(path):
            raise IOError('Config file not found: %s\nPerhaps you used --resume incorrectly?' % path)
        config = Config(path)
    
        # parameters of the instrumental distribution
        proxy_weights = scaled_fmap(config.items('proxy'))

        # parameters of the target distribution
        target_weights = scaled_fmap(config.items('target'))

        return JointWMap(WMap(sorted(proxy_weights.iteritems(), key=lambda (k, v): k)),
                WMap(sorted(target_weights.iteritems(), key=lambda (k, v): k)))

    def get_nsamples(self, iteration):
        if len(self.sampling_schedule) == 0:  # nothing in the schedule
            return 1000  # we resort to a default parameter
        elif len(self.sampling_schedule) == 1:  # a single value means keep on sampling this many derivations
            return self.sampling_schedule[0]
        else: # two or more
            # the first parameter is the number of samples, the second how many iterations 
            if self.sampling_schedule[1] > 0:   # for as long as we are meant to sample
                if iteration > self.sampling_schedule_iteration:  # this is a new iteration
                    self.sampling_schedule_iteration = iteration  # update progress
                    self.sampling_schedule[1] -= 1                # discount one iteration
                return self.sampling_schedule[0] 
            else:  # times is zero
                # remove the exhausted configuration from the schedule
                self.sampling_schedule.popleft()
                self.sampling_schedule.popleft()
                return self.get_nsamples(iteration) # and try again
    
    def optimise_target(self, run, devset, S, L, Tp, l2_weight):
                                                
        def f(theta):

            run.next_targetsgd()
            
            self.wmap.target.update(theta)
            logging.info('[%s] theta=%s', run, npvec2str(theta))

            R = []
            dR = []
            H = []
            dH = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                lmap = L[i]
                local_R, local_dR, local_H, local_dH = optalg.minrisk(derivations, 
                        q_wmap=self.wmap.proxy, 
                        p_wmap=self.wmap.target, 
                        lmap=lmap,
                        empirical_q=True)

                R.append(local_R)
                dR.append(local_dR)
                H.append(local_H)
                dH.append(local_dH)

            obj, jac = np.mean(R, 0), np.mean(dR, 0)

            if Tp == 0.0 and l2_weight == 0.0:
                logging.info('[%s] Risk=%f', run, obj)
                return obj, jac
            else:
                r_obj = obj
                r_jac = jac.copy()

                if Tp != 0.0:  # H-regularised
                    r_obj -= Tp * np.mean(H, 0)
                    r_jac -= Tp * np.mean(dH, 0)
                    logging.info('[%s] Risk=%f H-regularised=%f', run, obj, r_obj)

                if l2_weight != 0.0:  # L2-regularised
                    regulariser = LA.norm(theta, 2) ** 2
                    r_obj += l2_weight * regulariser
                    r_jac += 2 * l2_weight * theta
                    logging.info('[%s] Risk=%f L2-regularised=%f', run, obj, r_obj)
                
                return r_obj, r_jac

        def callback(theta):
            logging.info('[%s] new theta: %s', run, npvec2str(theta))

        t0 = time()
        logging.info('[%s] Minimising risk', run)
        result = minimize(f, 
                self.wmap.target.asarray, 
                #method='BFGS', 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxiter': self.args.psgd[0],
                    'ftol': self.args.ptol[0],
                    'gtol': self.args.ptol[1],
                    'maxfun': self.args.psgd[1],
                    'disp': False})
        dt = time() - t0
        logging.info('[%d] Target SGD: function=%f nfev=%d nit=%d success=%s message="%s" minutes=%s', run.iteration, 
                result.fun, result.nfev, result.nit, result.success, result.message, dt/60)
        return result.x
    
    def optimise_proxy(self, run, devset, S, L, Tq, l2_weight, N):
        
        polarity = 1.0
        if self.args.qopt == 'minkl':
            logging.info('[%s] Minimising KL divergence', run)
            get_obj_and_jac = optalg.minkl
        elif self.args.qopt == 'maxelb':
            logging.info('[%s] Maximising ELB', run)
            get_obj_and_jac = optalg.maxelb
            polarity = -1.0
        elif self.args.qopt == 'minvar':
            logging.info('[%s] Minimising variance of IS', run)
            get_obj_and_jac = optalg.minvar
        else:
            raise NotImplementedError('Unsupported optimisation method: %s', self.args.qopt)
                                                
        def f(theta):

            run.next_proxysgd()
            self.wmap.proxy.update(theta)
            logging.info('[%s] lambda=%s', run, npvec2str(theta))

            OBJ = []
            JAC = []
            H = []
            dH = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                lmap = L[i]
                local_obj, local_jac, local_H, local_dH = get_obj_and_jac(derivations, 
                        q_wmap=self.wmap.proxy, 
                        p_wmap=self.wmap.target, 
                        lmap=lmap,
                        empirical_q=False)
                
                OBJ.append(local_obj)
                JAC.append(local_jac)
                H.append(local_H)
                dH.append(local_dH)

            obj, jac = np.mean(OBJ, 0), np.mean(JAC, 0)
            if polarity != 1:
                obj *= polarity
                jac *= polarity

            if Tq == 0.0 and l2_weight == 0.0:
                logging.info('[%s] O=%f', run, obj)
                return obj, jac
            else:
                r_obj = obj
                r_jac = jac.copy()

                if Tq != 0.0:
                    r_obj -= Tq * np.mean(H, 0)
                    r_jac -= Tq * np.mean(dH, 0)
                    logging.info('[%s] O=%f H-regularised=%f', run, obj, r_obj)

                if l2_weight != 0.0:  # regularised
                    regulariser = LA.norm(theta, 2) ** 2
                    r_obj += l2_weight * regulariser
                    r_jac += 2 * l2_weight * theta
                    logging.info('[%s] O=%f L2-regularised=%f', run, obj, r_obj)
                return r_obj, r_jac
            

        def callback(theta):
            logging.info('[%s] new lambda: %s', run, npvec2str(theta))

        t0 = time()
        result = minimize(f, 
                self.wmap.proxy.asarray, 
                #method='BFGS', 
                method='L-BFGS-B', 
                jac=True, 
                callback=callback, 
                options={'maxiter': self.args.qsgd[0],
                    'ftol': self.args.qtol[0],
                    'gtol': self.args.qtol[1],
                    'maxfun': self.args.qsgd[1],
                    'disp': False})
        dt = time() - t0
        logging.info('[%d] Proxy SGD: function=%f nfev=%d nit=%d success=%s message="%s" minutes=%s', run.iteration,  
                result.fun, result.nfev, result.nit, result.success, result.message, dt/60)
        return result.x


    @staticmethod
    def _START_MODEL_(config, default=None, overwrite_q=False):
        """
        Requires: config, config_learning
        Produces: wmap
        """
        # parameters of the instrumental distribution
        proxy_weights = scaled_fmap(config.items('proxy'))
        if default is not None:
            proxy_weights = {k: default for k, v in proxy_weights.iteritems()}

        # parameters of the target distribution
        target_weights = scaled_fmap(config.items('target'))
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
        config.remove_section('proxy')
        config.add_section('proxy')
        [config.set('proxy', f, v) for f, v in proxy_wmap.iteritems()]
        
        config.remove_section('target')
        config.add_section('target')
        [config.set('target', f, v) for f, v in target_wmap.iteritems()]
        
        with smart_wopen('{0}/config0.ini'.format(workspace)) as fo:
            config.write(fo)

        return '{0}/config0.ini'.format(workspace)

    @staticmethod
    def _PREPARE_DEVSET_(workspace, path, config, alias='dev', input_format='cdec', grammar_dir=None):
        # load dev set and separate input and references
        logging.info('Reading %s set: %s', alias, path)
        if grammar_dir is None:
            if config.has_section('chisel:sampler'):
                sampler_map = dict(config.items('chisel:sampler'))
                grammar_dir = sampler_map.get('grammars', None)
        with smart_ropen(path) as f:
            devset = [SegmentMetaData.parse(line.strip(),
                                              input_format,
                                              grammar_dir=grammar_dir)
                        for sid, line in enumerate(f)]
        logging.info('%d %s instances', len(devset), alias)

        # dump source and references
        with smart_wopen('{0}/{1}.input'.format(workspace, alias)) as fi:
            with smart_wopen('{0}/{1}.refs'.format(workspace, alias)) as fr:
                for seg in devset:
                    print >> fi, seg.to_sgm(dump_refs=False)
                    print >> fr, ' ||| '.join(str(ref) for ref in seg.refs)
        return devset
    
    @staticmethod
    def _LOAD_METRIC_(config, metric):
        # loads mteval modules
        if config.has_section('chisel:metrics'):
            metrics_map = dict(config.items('chisel:metrics'))
        else:
            metrics_map = {'bleu': 'chisel.mteval.bleu'}
        mteval.load(metrics_map, frozenset([metric]))

        if not mteval.sanity_check(metric):
            raise Exception("Perhaps you forgot to include the metric '%s' in the configuration file?" % metric)

        # configure mteval metrics
        if config.has_section('chisel:metrics:config'):
            metrics_config = dict(config.items('chisel:metrics:config'))
        else:
            metrics_config = {}
        logging.debug('chisel:metrics:config: %s', metrics_config)
        mteval.configure(metrics_config)

    def devtest_eval(self, iteration, samples):
        """
        Evaluation of devtest set always happens at end of iteration, thus we use the same config file.
        """
        self.sample('run{0}'.format(iteration), 
                self.args.devtest_alias, 
                config='config{0}.ini'.format(iteration), 
                samples=samples, 
                grammar=self.args.devtest_grammar)
        self.decide('run{0}'.format(iteration), 
                self.args.devtest_alias, 
                config='config{0}.ini'.format(iteration))
        return self.assess('run{0}'.format(iteration), self.args.devtest_alias)

    def sample(self, run, alias, config, samples=1000, grammar=None, extra_parameters=''):
        """
        Sample derivation for a certain set of segments.

        :param workspace: workspace
        :param alias: alias of the set (determines the workspace)
        :param config: the number of the configuration file to be used (if not given, we assume the same as iteration)
            For example, sample(1, 'dev', 0)  will sample at the beginning of iteration 1 using config0.ini.
            Alternatively, sample(1, 'devtest', 1) will sample at the end of iteration 1 using config1.ini.
        :param samples: how samples to draw
        :param grammar: path to a grammar (typically necessary when sampling for a devtest set)
        :param extra_parameters: additional parameters to chisel.sampler
        :returns: path to samples
        """
        options = {'config': '{0}/{1}'.format(self.workspace, config), 
                'workspace': '{0}/{1}/{2}'.format(self.workspace, run, alias),
                'samples': samples}
        mkdir(options['workspace'])
        # command line
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s --samples %(samples)d' % options
        # additional parameters including --grammar
        if grammar is not None:
            cmd_str = '{0} --grammar {1}'.format(cmd_str, grammar)
        if extra_parameters:
            cmd_str = '{0} {1}'.format(cmd_str, extra_parameters)
        logging.debug('[%s] Run: %s', run, cmd_str)
        # prepare args
        cmd_args = shlex.split(cmd_str)
        # sample
        t0 = time()
        logging.info('[%s] Sampling %d solutions (%s)...', run, samples, alias)
        with smart_ropen('{0}/{1}.input'.format(self.workspace, alias)) as fi:
            with smart_wopen(self.path_to_log('sampling-{0}'.format(alias), run)) as fo:
                with smart_wopen(self.path_to_log('sampling-{0}'.format(alias), run, err=True)) as fe:
                    fe.write('{0}\n'.format(cmd_str))
                    proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                    proc.wait()
        dt = time() - t0
        logging.info('[%s]  sampling took %f seconds', run, dt)
        return '{0}/samples'.format(options['workspace'])
    
    def decide(self, run, alias, config, extra_parameters=''):
        """
        Apply a decision rule..

        :param iteration: current iteration (determines the run folder)
        :param alias: alias of the set (determines the workspace)
        :param config: the number of the configuration file to be used (if not given, we assume the same as iteration)
            For example, decide(1, 'dev', 0)  will decide from samples drawn at the beginning of iteration 1 using config0.ini.
            Alternatively, decide(1, 'devtest', 1) will decide from sample drawn at the end of iteration 1 using config1.ini.
        :param extra_parameters: additional parameters to chisel.fast_consensus
        :returns: (path to ranked decisions, path to 1-best outputs)
        """
        # required options
        options = {'config': '{0}/{1}'.format(self.workspace, config), 
                'workspace': '{0}/{1}/{2}'.format(self.workspace, run, alias)}
        # command line
        cmd_str = 'python -m chisel.fast_consensus %(config)s %(workspace)s ' % options
        # additional parameters
        if extra_parameters:
            cmd_str = '{0} {1}'.format(cmd_str, extra_parameters)
        logging.debug('[%s] Run: %s', run, cmd_str)
        # perpare args
        cmd_args = shlex.split(cmd_str)
        # decide
        t0 = time()
        logging.info('[%s] Deciding (%s)...', run, alias)
        with smart_wopen(self.path_to_log('decision-{0}'.format(alias), run)) as fo:
            with smart_wopen(self.path_to_log('decision-{0}'.format(alias), run, err=True)) as fe:
                proc = sp.Popen(cmd_args, stdin=None, stdout=fo, stderr=fe)
                proc.wait()
        dt = time() - t0
        logging.info('[%s]  deciding took %f seconds', run, dt)
        return '{0}/decisions'.format(options['workspace']), '{0}/output'.format(options['workspace'])
    
    def assess(self, run, alias):
        # where samples, decisions and outputs can be found
        workspace = '{0}/{1}/{2}'.format(self.workspace, run, alias)
        # command line
        cmd_str = '{0} -r {1}'.format(self.args.scoring_tool, '{0}/{1}.refs'.format(self.workspace, alias))
        logging.debug('[%s] Run: %s', run, cmd_str)
        # prepare args
        cmd_args = shlex.split(cmd_str)
        # assess
        t0 = time()
        trans_path = '{0}/output/consensus-bleu'.format(workspace)
        logging.info('[%s] Assessing (%s)...', run, alias)
        score = None
        with smart_ropen(trans_path) as fin:
            bleu_out = '{0}.bleu.stdout'.format(splitext(trans_path)[0])
            bleu_err = '{0}.bleu.stderr'.format(splitext(trans_path)[0])
            with smart_wopen(bleu_out) as fout:
                with smart_wopen(bleu_err) as ferr:
                    # logging.info(cmd_args)
                    proc = sp.Popen(cmd_args, stdin=fin, stdout=fout, stderr=ferr)
                    proc.wait()
                    try:
                        with smart_ropen(bleu_out) as fi:
                            line = next(fi)
                            score = float(line.strip())
                    except:
                        logging.error('[%s] Problem reading %s for %s', run, bleu_out, alias)
        dt = time() - t0
        logging.info('[%s]  assessing took %f seconds', run, dt)
        return score

    def training_loss(self, run, alias, segments, samples):
        L = []

        if self.args.save_loss:
            loss_dir = '{0}/{1}/loss'.format(self.workspace, run, alias)
            mkdir(loss_dir)

        logging.info('[%s] Computing loss (%s)...', run, alias)
        t0 = time()
        # run fast bleu implementation
        # TODO: generalise to other metrics
        for seg, derivations in zip(segments, samples):
            projections = frozenset(d.tree.projection for d in derivations)
            scorer = TrainingBLEU(seg.refs)
            lmap = {y: scorer.loss(y.split()) for y in projections}
            L.append(lmap)
            if self.args.save_loss:
                with smart_wopen('{0}/{1}.gz'.format(loss_dir, seg.id)) as fo:
                    for d in derivations:
                        fo.write('{0}\n'.format(lmap[d.tree.projection]))
        dt = time() - t0
        logging.info('[%s]  computing loos took %s seconds', run, dt)
        return L

    def read_samples(self, run, alias):
        samples_dir = '{0}/{1}/{2}/samples'.format(self.workspace, run, alias)
        logging.info('[%s] Reading samples (%s)...', run, alias)
        input_files = list_numbered_files(samples_dir)
        S = []
        t0 = time()
        for fid, input_file in input_files:
            logging.debug(' reading %s from %s', fid, input_file)
            derivations, _qmap, _pmap = sampled_derivations_from_file(input_file)
            S.append(derivations)
        dt = time() - t0
        logging.info('[%s]  reading samples took %f seconds', run, dt)
        return S
    
    
    def tune(self):

        metric = self.args.metric
        devset = self.devset
    
        run = RunCount()
        Tp = self.args.Tp
        Tq = self.args.Tq
        pcooling = self.args.pcooling_lag
        qcooling = self.args.qcooling_lag
        
        # Iteration 0 consists in assessing devtest with the initial weight vector
        if self.args.resume == 0:
            devtest_score = self.devtest_eval(run.iteration, samples=self.get_nsamples(0))
            logging.info('[%d] Devtest eval (initialiser): %s', run.iteration, devtest_score)

        for loop in range(1, self.args.maxiter + 1):  
            run.next_run()
            t0 = time()
            # deal with sampling schedule
            N = self.get_nsamples(run.iteration)
            # deal with cooling schedule
            if pcooling <= 0:
                Tp /= self.args.pcooling_factor  # cool down
                if Tp < self.args.minTp:  # ensure minimum temperature
                    Tp = self.args.minTp
                pcooling = self.args.pcooling_lag  # reset lag
            pcooling -= 1
            if qcooling <= 0:
                Tq /= self.args.qcooling_factor  # cool down
                if Tq < self.args.minTq:  # ensure minimum temperature
                    Tq = self.args.minTq  
                qcooling = self.args.qcooling_lag  # reset lag
            qcooling -= 1
            logging.info('[%d] Tp=%s Tq=%s', run.iteration, Tp, Tq)
            # skip complete iterations
            if loop < self.args.resume:
                continue
            path_to_run = '{0}/run{1}'.format(self.workspace, run.iteration)
            mkdir(path_to_run)

            # prepare a config file within run
            self.update_config_file('config{0}.ini'.format(run.iteration - 1), 
                    'run{0}/0.ini'.format(run.iteration))
            # optimise coordinates
            if self.args.order == 'pq':
                S, L = self.coordinate_p(devset, run, shift=0, resample=True, N=N, Tp=Tp)
                # we shift as many iterations as we had to run for P
                self.coordinate_q(devset, run, shift=self.args.piter, resample=self.args.resample, N=N, Tq=Tq, S=S, L=L)
            else:  # qp
                S, L = self.coordinate_q(devset, run, shift=0, resample=True, N=N, Tq=Tq)
                # we shift as many iterations as we had to run for Q
                self.coordinate_p(devset, run, shift=self.args.qiter, resample=self.args.resample, N=N, Tp=Tp, S=S, L=L)
            
            # update the config that precedes this run
            self.update_config_file('run{0}/{1}.ini'.format(run.iteration, self.args.piter + self.args.qiter), 
                    'config{0}.ini'.format(run.iteration))
            devtest_score = self.devtest_eval(run.iteration, samples=N)
            logging.info('[%d] Devtest eval (end of iteration): %s', run.iteration, devtest_score)
            dt = time() - t0
            logging.info('[%d] Iteration took %s minutes', run.iteration, dt/60)

    def coordinate_p(self, devset, run, shift, resample, N, Tp, S=None, L=None):
        for i in range(1, self.args.piter + 1):
            run.next_target()
            # sample for dev (using the previous config file)
            config = 'run{0}/{1}.ini'.format(run.iteration, i + shift - 1)
            curr = 'run{0}/{1}'.format(run.iteration, i + shift)
            if resample or i > 1:
                samples_dir = self.sample(curr, self.args.dev_alias, config=config, samples=N)
                if not os.path.isdir(samples_dir):
                    raise Exception('[%s] could not find samples' % curr)
                # eval dev set if necessary
                if not self.args.no_eval_dev:
                    self.decide(curr, self.args.dev_alias, config=config)
                    dev_score = self.assess(curr, self.args.dev_alias)
                    logging.info('[%s] Dev eval (begin of iteration): %s', curr, dev_score)
                # load samples for optimisation
                S = self.read_samples(curr, self.args.dev_alias)
                # compute loss
                L = self.training_loss(curr, self.args.dev_alias, segments=devset, samples=S)
            # SGD
            target_weights = self.optimise_target(run, devset, S, L, Tp, self.args.pL2)
            self.wmap.target.update(target_weights)
            self.update_config_file(config, 'run{0}/{1}.ini'.format(run.iteration, i + shift))
        return S, L

    def coordinate_q(self, devset, run, shift, resample, N, Tq, S=None, L=None):
        for i in range(1, self.args.qiter + 1):
            run.next_proxy()
            # sample for dev (using the previous config file)
            config = 'run{0}/{1}.ini'.format(run.iteration, i + shift - 1)
            curr = 'run{0}/{1}'.format(run.iteration, i + shift)
            if resample or i > 1:  # first itereation reuses from p unless the user wants to resample
                samples_dir = self.sample(curr, self.args.dev_alias, config=config, samples=N)
                if not os.path.isdir(samples_dir):
                    raise Exception('[%s] could not find samples' % curr)
                # eval dev set if necessary
                if not self.args.no_eval_dev:
                    self.decide(curr, self.args.dev_alias, config=config)
                    dev_score = self.assess(curr, self.args.dev_alias)
                    logging.info('[%s] Dev eval (begin of iteration): %s', curr, dev_score)
                # load samples for optimisation
                S = self.read_samples(curr, self.args.dev_alias)
                # compute loss
                L = self.training_loss(curr, self.args.dev_alias, segments=devset, samples=S)
            # SGD
            proxy_weights = self.optimise_proxy(run, devset, S, L, Tq, self.args.qL2, N)
            self.wmap.proxy.update(proxy_weights)
            self.update_config_file(config, 'run{0}/{1}.ini'.format(run.iteration, i + shift))
        return S, L
