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
        self.sampling_schedule = deque(self.args.samples)
        self.sampling_schedule_iteration = 0
    
        self.alpha_p = 1.0 # 0.1
        self.alpha_q = 1.0 # 0.1
        self.workspace = Driver._MAKE_WORKSPACE_(self.args.workspace, 'SGD', self.args.metric, self.args.alias)

        if args.resume == 0:
            self.wmap = Driver._START_MODEL_(self.config, default=self.args.default)
            Driver._BASE_CONFIG_(self.config, self.workspace, self.wmap.proxy, self.wmap.target)
        else:
            logging.info('Loading model %s', '{0}/config{1}.ini'.format(self.workspace, args.resume - 1))
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
        return '{0}/log/{1}.{2}.std{3}'.format(self.workspace, source, iteration, 'err' if err else 'out')
    
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
                    # logging.info(cmd_args)
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
            # sample for dev (using the previous config file)
            samples_dir = self.sample(run.iteration, self.args.dev_alias, config=run.iteration - 1, samples=N)  
            if not os.path.isdir(samples_dir):
                raise Exception('[%d] could not find samples' % run.iteration)
            # eval dev set if necessary
            if not self.args.no_eval_dev:
                self.decide(run.iteration, self.args.dev_alias, config=run.iteration - 1)
                dev_score = self.assess(run.iteration, self.args.dev_alias)
                logging.info('[%d] Dev eval (begin of iteration): %s', run.iteration, dev_score)
            # load samples for optimisation
            S = self.read_samples(run.iteration, self.args.dev_alias)
            # compute loss
            L = self.training_loss(run.iteration, self.args.dev_alias, segments=devset, samples=S)
            # tuning iteration
            if self.args.order == 'pq':
                target_weights = self.optimise_target(run, devset, S, L, Tp, self.args.pL2)
                self.wmap.target.update(target_weights)
                proxy_weights = self.optimise_proxy(run, devset, S, Tq, self.args.qL2)
                self.wmap.proxy.update(proxy_weights)
            else:
                proxy_weights = self.optimise_proxy(run, devset, S, Tq, self.args.qL2)
                self.wmap.proxy.update(proxy_weights)
                target_weights = self.optimise_target(run, devset, S, L, Tp, self.args.pL2)
                self.wmap.target.update(target_weights)
            # update the config that precedes this run
            self.update_config_file(run.iteration - 1)
            devtest_score = self.devtest_eval(run.iteration, samples=N)
            logging.info('[%d] Devtest eval (end of iteration): %s', run.iteration, devtest_score)
            dt = time() - t0
            logging.info('[%d] Iteration took %s minutes', run.iteration, dt/60)


    def optimise_target(self, run, devset, S, L, Tp, l2_weight):
                                                
        def f(theta):

            run.next_risk()
            
            self.wmap.target.update(theta)
            logging.info('[%s] theta=%s', run, npvec2str(theta))

            emp_risk = []
            emp_dR = []
            H = []
            dH = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                lmap = L[i]
                #empdist = EmpiricalDistribution(derivations,
                #                                q_wmap=self.wmap.proxy,
                #                                p_wmap=self.wmap.target,
                #                                empirical_q=True,  # crucial: the proposal is fixed, thus we can rely on empirical estimates
                #                                get_yield=lambda d: d.tree.projection)
                support, posterior, dP, local_H, local_dH = optalg.minrisk(derivations, 
                        q_wmap=self.wmap.proxy, p_wmap=self.wmap.target, 
                        empirical_q=True, get_yield=lambda d: d.tree.projection)

                losses = np.array([lmap[Dy.projection] for Dy in support], float)  # l(y)
                #posterior = empdist.copy_posterior()  # p(y)
                #dP = empdist.copy_dpdt() #  dp(y)/dt
                dR = losses.dot(dP) 
                risk = losses.dot(posterior.transpose())
                emp_risk.append(risk)
                emp_dR.append(dR)
                H.append(local_H)
                dH.append(local_dH)

            obj, jac = np.mean(emp_risk, 0), np.mean(emp_dR, 0)

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
                    regulariser = LA.norm(theta, 2)
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
    
    def optimise_proxy(self, run, devset, S, Tq, l2_weight):
        
        if self.args.qopt == 'minkl':
            logging.info('[%s] Minimising KL divergence', run)
            get_obj_and_jac = optalg.minkl
            polarity = 1
        elif self.args.qopt == 'maxelb':
            logging.info('[%s] Maximising ELB', run)
            get_obj_and_jac = optalg.maxelb
            polarity = -1
        else:
            raise NotImplementedError('Unsupported optimisation method: %s', self.args.qopt)
        #elif self.args == 'minvar': 
        #    pass

                                                
        def f(theta):

            run.next_kl()
            self.wmap.proxy.update(theta)
            logging.info('[%s] lambda=%s', run, npvec2str(theta))

            OBJ = []
            JAC = []
            H = []
            dH = []
            for i, seg in enumerate(devset):
                derivations = S[i]
                #empdist = EmpiricalDistribution(derivations,
                #                                q_wmap=self.wmap.proxy,
                #                                p_wmap=self.wmap.target,
                #                                empirical_q=False,  # crucial: the proposal is changing, thus we cannot rely on empirical estimates
                #                                get_yield=lambda d: d.tree.projection)

                local_obj, local_jac, local_H, local_dH = get_obj_and_jac(derivations, q_wmap=self.wmap.proxy, p_wmap=self.wmap.target, empirical_q=False)
                
                #elb, delb = empdist.elb()
                #kl, dkl = empdist.kl()
                # TODO: min KL  or min - ELB (should be equivalent, shouldn't it?)
                OBJ.append(local_obj)
                JAC.append(local_jac)
                H.append(local_H)
                dH.append(local_dH)
                #hq, dhq = empdist.Hq()
                #H.append(hq)
                #dH.append(dhq)

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
                    regulariser = LA.norm(theta, 2)
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
    def _PREPARE_DEVSET_(workspace, path, config, alias='dev', input_format='cdec', grammar_dir=None):
        # load dev set and separate input and references
        logging.info('Reading %s set: %s', alias, path)
        if grammar_dir is None:
            if config.has_section('chisel:sampler'):
                sampler_map = section_literal_eval(config.items('chisel:sampler'))
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

    def devtest_eval(self, iteration, samples):
        """
        Evaluation of devtest set always happens at end of iteration, thus we use the same config file.
        """
        self.sample(iteration, self.args.devtest_alias, config=iteration, samples=samples, grammar=self.args.devtest_grammar)
        self.decide(iteration, self.args.devtest_alias, config=iteration)
        return self.assess(iteration, self.args.devtest_alias)

    def dev_eval(self, iteration):
        """
        Evaluation of dev set always happens at the beginning of iteration, thus we use the preceding config file.
        """
        self.decide(iteration, self.args.dev_alias, config=iteration - 1)
        return self.assess(iteration, self.args.dev_alias)
    
    def sample(self, iteration, alias, config=None, samples=1000, grammar=None, extra_parameters=''):
        """
        Sample derivation for a certain set of segments.

        :param iteration: current iteration (determines the run folder)
        :param alias: alias of the set (determines the workspace)
        :param config: the number of the configuration file to be used (if not given, we assume the same as iteration)
            For example, sample(1, 'dev', 0)  will sample at the beginning of iteration 1 using config0.ini.
            Alternatively, sample(1, 'devtest', 1) will sample at the end of iteration 1 using config1.ini.
        :param samples: how samples to draw
        :param grammar: path to a grammar (typically necessary when sampling for a devtest set)
        :param extra_parameters: additional parameters to chisel.sampler
        :returns: path to samples
        """
        # required options
        if config is None:
            config = iteration
        options = {'config': '{0}/config{1}.ini'.format(self.workspace, config), 
                'workspace': '{0}/run{1}/{2}'.format(self.workspace, iteration, alias),
                'samples': samples}
        mkdir(options['workspace'])
        # command line
        cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s --samples %(samples)d' % options
        # additional parameters including --grammar
        if grammar is not None:
            cmd_str = '{0} --grammar {1}'.format(cmd_str, grammar)
        if extra_parameters:
            cmd_str = '{0} {1}'.format(cmd_str, extra_parameters)
        logging.debug('[%d] Run: %s', iteration, cmd_str)
        # prepare args
        cmd_args = shlex.split(cmd_str)
        # sample
        t0 = time()
        logging.info('[%d] Sampling %d solutions (%s)...', iteration, samples, alias)
        with smart_ropen('{0}/{1}.input'.format(self.workspace, alias)) as fi:
            with smart_wopen(self.path_to_log('sampling-{0}'.format(alias), iteration)) as fo:
                with smart_wopen(self.path_to_log('sampling-{0}'.format(alias), iteration, err=True)) as fe:
                    fe.write('{0}\n'.format(cmd_str))
                    proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                    proc.wait()
        dt = time() - t0
        logging.info('[%d]  sampling took %f seconds', iteration, dt)
        #if not self.check_samples(self.iteration):
        #    raise Exception('chisel.sampler appears to have failed at iteration %d', run.iteration)
        return '{0}/samples'.format(options['workspace'])
    
    def decide(self, iteration, alias, config=None, extra_parameters=''):
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
        if config is None:
            config = iteration
        # required options
        options = {'config': '{0}/config{1}.ini'.format(self.workspace, config), 
                'workspace': '{0}/run{1}/{2}'.format(self.workspace, iteration, alias)}
        # command line
        cmd_str = 'python -m chisel.fast_consensus %(config)s %(workspace)s ' % options
        # additional parameters
        if extra_parameters:
            cmd_str = '{0} {1}'.format(cmd_str, extra_parameters)
        logging.debug('[%d] Run: %s', iteration, cmd_str)
        # perpare args
        cmd_args = shlex.split(cmd_str)
        # decide
        t0 = time()
        logging.info('[%d] Deciding (%s)...', iteration, alias)
        with smart_wopen(self.path_to_log('decision-{0}'.format(alias), iteration)) as fo:
            with smart_wopen(self.path_to_log('decision-{0}'.format(alias), iteration, err=True)) as fe:
                proc = sp.Popen(cmd_args, stdin=None, stdout=fo, stderr=fe)
                proc.wait()
        dt = time() - t0
        logging.info('[%d]  deciding took %f seconds', iteration, dt)
        return '{0}/decisions'.format(options['workspace']), '{0}/output'.format(options['workspace'])
    
    def assess(self, iteration, alias):
        # where samples, decisions and outputs can be found
        workspace = '{0}/run{1}/{2}'.format(self.workspace, iteration, alias)
        # command line
        cmd_str = '{0} -r {1}'.format(self.args.scoring_tool, '{0}/{1}.refs'.format(self.workspace, alias))
        logging.debug('[%d] Run: %s', iteration, cmd_str)
        # prepare args
        cmd_args = shlex.split(cmd_str)
        # assess
        t0 = time()
        trans_path = '{0}/output/consensus-bleu'.format(workspace)
        logging.info('[%d] Assessing (%s)...', iteration, alias)
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
                        logging.error('[%d] Problem reading %s for %s', iteration, bleu_out, alias)
        dt = time() - t0
        logging.info('[%d]  assessing took %f seconds', iteration, dt)
        return score

    def training_loss(self, iteration, alias, segments, samples):
        L = []
        loss_dir = '{0}/run{1}/loss'.format(iteration, alias)
        mkdir(loss_dir)

        logging.info('[%d] Computing loss (%s)...', iteration, alias)
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
        logging.info('[%d]  computing loos took %s seconds', iteration, dt)
        return L

    def read_samples(self, iteration, alias):
        samples_dir = '{0}/run{1}/{2}/samples'.format(self.workspace, iteration, alias)
        logging.info('[%d] Reading samples (%s)...', iteration, alias)
        input_files = list_numbered_files(samples_dir)
        S = []
        t0 = time()
        for fid, input_file in input_files:
            logging.debug(' reading %s from %s', fid, input_file)
            derivations, _qmap, _pmap = sampled_derivations_from_file(input_file)
            S.append(derivations)
        dt = time() - t0
        logging.info('[%d]  reading samples took %f seconds', iteration, dt)
        return S
