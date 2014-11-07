"""
Minimum Bayes Risk decoding for SMT (Kumar and Byrne, 2003):

        c^ = \argmin_c \sum_r L_r(c)p(r)

This decision rule runs in time O(|C||R|).
DeNero et al (2009) introduce Consensuns Decoding, which is basically MBR over feature representations:

    Assume the loss a function of the form:

        L_r(c) = \sum_j \omega_j(c) \phi_j(r)

    Then the risk becomes:

        R(c)    = \sum_r L_r(c)p(r)
                = \sum_r p(r) \sum_j \omega_j(c) \phi_j(r)
                = \sum_j \omega_j(c) \sum_r \phi_j(r) p(r)
                = \sum_j \omega_j(c) E_p[\phi_j(r)]

    And the decision rule becomes:

        c^ = \argmin_c \sum_j \omega_j(c) E_p[\phi_j(r)]

    Note that this rule runs in time O(|C|k) where k is the number of features.

Consensus Training (Pauls et al, 2009):

        w^ = \argmin_w \sum_{(x,r) \in D} \sum_c L_r(c)p(c|x; w)

    Remark: for fixed parameters w, the number of evaluations of the loss is proportional to O(|D||C|).

    Again, assume the loss a function of the form:

        L_r(c) = \sum_j \omega_j(c) \phi_j(r)

    Then the risk becomes (omitting the dependency on x and w):

        R(c)    = \sum_c L_r(c)p(c)
                = \sum_c p(c) \sum_j \omega_j(c) \phi_j(r)
                = \sum_j \phi_j(r) \sum_c \omega_j(c) p(c)
                = \sum_j \phi_j(r) E_p[\omega_j(c)]

    Ant the objective becomes

        w^ = \argmin_w \sum_{(x,r) \in D} \sum_j \phi_j(r) E_p[\omega_j(c)]

    Remark: now, for fixed parameters w, the number of evaluations of the loss is proportional to O(|C|k)

@author waziz
"""
import chisel.mteval as mteval


def consensus(E, metric, normalise=False):
    scores = [mteval.comparison(c=i, r=mteval.EXPECTED, metric=metric) for i, Dy in enumerate(E)]
    return scores