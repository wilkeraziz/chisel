import numpy as np
import itertools
import sys
import math
import logging
import chisel.mteval as mteval
from collections import defaultdict
from time import time
from chisel.util.iotools import sampled_derivations_from_file
from chisel.decoder.estimates import EmpiricalDistribution
from chisel.smt import groupby
from collections import namedtuple
from chisel.util import npvec2str


RiskReturn = namedtuple('RiskReturn', 'R dR KL')
DivergenceReturn = namedtuple('DivergenceReturn', 'KL dKL')

def save_estimates(sid, empdist, losses, posterior, risk, dR, stems):
    """

    """

    with open('{0}/{1}'.format(stems[0], sid), 'w') as fo:
        print >> fo, empdist
    
    with open('{0}/{1}'.format(stems[1], sid), 'w') as fo:
        print >> fo, '#loss'
        print >> fo, '\n'.join('%.4f' % l for l in losses)
    
    with open('{0}/{1}'.format(stems[2], sid), 'w') as fo:
        print >> fo, '#risk\t#dR'
        print >> fo, '{0}\t{1}'.format(risk, npvec2str(dR)) 
    

def risk(seg, samples_file, q_wmap, p_wmap, metric, sample_headers, consensus=False, save_to=None):
    # TODO: log information in file $workspace/estimates/$seg
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    derivations, _qmap, _pmap = sampled_derivations_from_file(samples_file) #, sample_headers)
    empdist = EmpiricalDistribution(derivations,
                                    q_wmap=q_wmap,
                                    p_wmap=p_wmap,
                                    get_yield=lambda d: d.tree.projection)

    mteval.prepare_training(seg.src, seg.refs, empdist)
    if consensus:
        raise Exception('Consensus training is not yet supported.')
    else:
        M = len(empdist)
        losses = np.array([mteval.training_loss(c=h, metric=metric) for h, hyp in enumerate(empdist)])
        posterior = empdist.copy_posterior()
        dP = empdist.copy_dpdt() 
        dR = losses.dot(dP)
        risk = losses.dot(posterior.transpose())
    
    if save_to is not None:
        save_estimates(seg.id, empdist, losses, posterior, risk, dR, save_to)
    
    kl, dkl = empdist.kl()
    return RiskReturn(risk, dR, kl) 


def divergence(seg, samples_file, q_wmap, p_wmap, sample_headers, save_to=None):
    derivations, _qmap, _pmap = sampled_derivations_from_file(samples_file)  #, sample_headers)
    empdist = EmpiricalDistribution(derivations,
                                    q_wmap=q_wmap,
                                    p_wmap=p_wmap,
                                    get_yield=lambda d: d.tree.projection)

    kl, dkl = empdist.kl()
    return DivergenceReturn(kl, dkl)
