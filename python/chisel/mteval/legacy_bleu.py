"""
@author waziz
"""
import collections
import math
import numpy as np
#from scipy.sparse import csr_matrix, csc_matrix, lil_matrix


class NGram(object):
    """
    Represents the statistics associated with a certain ngram
    """

    def __init__(self, uid, words, d, posterior = 0.0):
        """
        @param uid the unique id of the ngram
        @param d dimensionality of the vector of counts (size of the evidence set)
        """
        self.id = uid
        self.words = words
        self.counts = np.zeros(d)
        self.posterior = 0

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __str__(self):
        return ' '.join(self.words)


class NGramFactory(object):

    def __init__(self, d):
        self.d_ = d
        self.ngrams_ = []
        self.n2i_ = collections.defaultdict()

    def get(self, words):
        """get(words) -> NGram object"""
        key = tuple(words)
        nid = self.n2i_.get(key, None)
        if nid is None:
            nid = len(self.ngrams_)
            self.ngrams_.append(NGram(nid, key, self.d_))
            self.n2i_[key] = nid
        return self.ngrams_[nid]

    def __getitem__(self, nid):
        """self[nid] -> NGram object"""
        return self.ngrams_[nid]
    
    def __iter__(self):
        """iterator over NGram objects"""
        return iter(self.ngrams_)

    def iteritems(self):
        """iterator over (words, NGram) tuples"""
        return ((words, self.ngrams_[nid]) for words, nid in self.n2i_.iteritems())


class CountByOrder(object):
    """
    Represents the statistics associated with each class of ngram.
    """

    def __init__(self, n):
        """
        @param n maximum ngram order
        """
        # cn[0] is the length
        # cn[k] k > 0 is the number of kgrams of order n
        self.cn = np.zeros(n + 1)


class EfficientClippedCounts(object):
    """
    Pre-computes the clipped counts between every two sentences.
    O(|E|^2*|I|) where E is the evidence set and I is the average hypothesis length.
    In comparison to InefficientClippedCounts this is faster because there is a direct map from samples to ngrams.
    """

    def __init__(self, snt2ngrams, n = 4):
        """
        Constructs the clipped counts from a list of dictionaries.
        Each element of the list corresponds to a sample i.
        The dictionary store key-value pairs of the type (ngram, occurrences in i).
        """
        # size of the evidence set
        M = len(snt2ngrams)

        # in self.cc_ there is a numpy array for each solution in the evidence set
        # each ci = self.cc_[i] has i+1 rows (one for each alternative solution) and n+1 columns (one for each ngram order)
        # each cell represents the clipped count for a certain ngram order between two solutions
        self.cc_ = [np.zeros((i + 1, n + 1)) for i in xrange(M)]
        # TODO: use cython prange
        for i in xrange(M):
            # cc_i[j][n] -> count_n(i, j)
            cci = self.cc_[i]  
            # count for columns (just the lower half of the matrix)
            for j in xrange(i + 1):
                # for each ngram order
                for k in range(1, n + 1):
                    # d1 is the smaller set, d2 is the larger one
                    (d1, d2) = (snt2ngrams[i][k], snt2ngrams[j][k]) if len(snt2ngrams[i][k]) <= len(snt2ngrams[j][k]) else (snt2ngrams[j][k], snt2ngrams[i][k])
                    # clip counts
                    cci[j,k] = sum(min(c1, d2.get(w, 0)) for w, c1 in d1.iteritems())

    def counts(self, i, j):
        """
        Returns a vector C such that C[k] is the clipped counts for kgrams between samples i and j
        """
        return self.cc_[i][j] if j < i else self.cc_[j][i]


class BLEUSufficientStatistics(object):

    def __init__(self, E, n=4):
        """
        @param E is an evidence set (where we gather ngram counts from) encoded as a list of Sample objects
        @param maximum ngram order (for standard BLEU this is 4)

        This computes:

        1) ngramstats: a list such that each element is associated with a kgram (k in [1,4]) and the value is an NGramStats object.
        Each NGramStats object contains a vector of M=len(E) counts representing the number of occurrences 
        of the ngram in each sample of the evidence set. It also contains the n-gram posterior probability computed
        as a function of normalised counts or scores.

        2) counstats: a list such that each element is a CountByOrder object.
        Each object summarises the counts for a certain kgram class (1 <= k <= n determines the class).

        3) snt2ngrams: a list of dictionaries, each dictionary is associated with a sample, its key-value pairs represent
        an ngram and how many times it occurs in the sample.
        This is somewhat redundant with (1) however it speeds up the computation of clipped counts (for exact BLEU).

        4) clippedcounts: an object of the type ClippedCounts which provides clipped counts (computed from (1) or (3) above).
        This object is lazily constructed.
        """

        M = len(E)
        self.evidence_ = E
        self.maxorder_ = n
        # 1) ngram count objects
        # ngram = ngram_counts[words] 
        # ngram.nid -> unique integer id
        # ngram.counts[i] -> number of times the ngram occurs in the i-th solution
        self.ngram_counts = NGramFactory(M)
        # 2) total counts by order
        # total_counts[i].cn[k] -> total number of k-grams in the i-th solution
        self.total_counts = [CountByOrder(n) for _ in xrange(M)]
        # 3) sentence-to-ngrams
        # snt2ngrams[i][k][nid] -> number of occurrences of nid (a k-gram), in the i-th solution
        self.snt2ngrams = [[collections.defaultdict(int) for _ in range(n + 1)] for _ in E]
        self._clippedcounts = None
        ngram_counts, total_counts, snt2ngrams = self.ngram_counts, self.total_counts, self.snt2ngrams
      
        # one CountByOrder object for each sample
        self.sample_count_by_order = [CountByOrder(n) for _ in E]

        # for each sample in the evidence set
        for i, sample in enumerate(E):
            # first count position is the length
            # the kth postion (1 <= k <= n) is associated with ngrams of length k
            total_counts[i].cn[0] = len(sample.leaves)
            # count kgrams for k=1..n
            context = collections.deque()
            for w in sample.leaves:
                # left trim the context
                if len(context) == n:
                    context.popleft()
                # count ngrams ending in w
                words = collections.deque([w])
                # a) process w
                ngram = ngram_counts.get(words)
                ngram.counts[i] += 1
                total_counts[i].cn[1] += 1
                snt2ngrams[i][1][ngram.id] += 1
                # b) process longer ngrams ending in w
                for h in context:
                    words.appendleft(h)
                    ngram = ngram_counts.get(words)
                    ngram.counts[i] += 1
                    total_counts[i].cn[len(words)] += 1
                    snt2ngrams[i][len(ngram)][ngram.id] += 1
                # c) adds w to the context 
                context.append(w)

        #self._total_sampled_derivations = sum(sum(derivation.count for derivation in sample) for sample in E)
        #N = float(self._total_sampled_derivations)
        # pre-computes ngram posteriors
        #for stats in ngram_counts:
        #    stats.posterior = sum(E.p(i)  for i, group in enumerate(E) if stats.counts[i] > 0)

    @property
    def evidence(self):
        return self.evidence_

    @property
    def maxorder(self):
        return self.maxorder_
    
    @property
    def clippedcounts(self):
        if self._clippedcounts is None:
            self._clippedcounts = EfficientClippedCounts(self.snt2ngrams, self.maxorder_)
        return self._clippedcounts

    def length(self, h):
        return self.total_counts[h].cn[0] # candidate length
    
    def tc(self, h):
        """
        Total counts in h for all orders
        """
        return self.total_counts[h].cn
                
    def tcn(self, h, n):
        """
        Total counts of order n in h
        """
        return self.total_counts[h].cn[n]
  
    def cc(self, h, r):
        """
        Return clipped counts between h and r for all orders.
        This assumes compute_clipped_counts was invoked.
        @return C such that C[k] is the clipped count for k-grams
        """
        return self.clippedcounts.counts(h, r)

    def ccn(self, h, r, n):
        """
        Clipped counts of order n between h and r
        This assumes compute_clipped_counts was invoked.
        @return clipped count
        """
        return self.clippedcounts.counts(h, r)[n]

    def ngrams(self, h, n):
        return self.snt2ngrams[h][n]


class CoBLEUSufficientStatistics(object):

    def __init__(self, bleusuff, normalise_posterior=False):
        E = bleusuff.evidence
        Len = np.matrix([bleusuff.length(i) for i in xrange(len(E))])
        Posterior = np.matrix([E.p(i, normalise_posterior) for i, Dy in enumerate(E)]).transpose()
        ExpLen = Len * Posterior
        Counts = np.array([ngram.counts for ngram in bleusuff.ngram_counts])
        ExpCounts = Counts * Posterior
        self.exp_length = ExpLen[0, 0]
        self.ExpCounts = ExpCounts


class BLEU(object):

    def __init__(self, evidence_set, impl='ibm_bleu', n=4):
        self._bleusuff = BLEUSufficientStatistics(evidence_set, n)
        self._cobleusuff = None
        self._impl = impl
        self._bleu = BLEU.get(impl)

    def bleusuff(self):
        return self._bleusuff
    
    def cobleusuff(self, normalise_posterior):
        if self._cobleusuff is None:
            self._cobleusuff = CoBLEUSufficientStatistics(self._bleusuff, normalise_posterior=normalise_posterior)
        return self._cobleusuff
        
    def exact(self, rid, hid):
        bleusuff = self.bleusuff()
        return self._bleu(r = bleusuff.length(rid), 
                c = bleusuff.length(hid), 
                cc = bleusuff.cc(hid, rid), 
                tc = bleusuff.tc(hid),
                n = bleusuff.maxorder)

    def expected(self, hid, normalise_posterior):
        # get sufficient statistics
        bleusuff = self.bleusuff()
        cobleusuff = self.cobleusuff(normalise_posterior)
        
        # clip to the expected counts
        cc = np.zeros(bleusuff.maxorder + 1)
        for k in range(1, bleusuff.maxorder + 1):
            for nid, count in bleusuff.ngrams(hid, k).iteritems():
                cc[k] += min(count, cobleusuff.ExpCounts[nid])

        # compute co-bleu
        b = self._bleu(r = cobleusuff.exp_length, 
                c = bleusuff.length(hid), 
                cc = cc,
                tc = bleusuff.tc(hid),
                n = bleusuff.maxorder)
        return b
    
    @classmethod
    def eval(cls, r, c, cc, tc, n, smoothing):
        """
        @param r reference length
        @param c candidate length
        @param cc (clipped counts) is a vector of clipped counts such that cc[k] is the count for k-grams
        @param tc (total counts) is a vector of the total ngram counts (for the candidate), tc[k] is the count for k-grams
        @param n max ngram order
        @param smoothing computes smoothed precisions from cc and tc (both adjusted to exactly n positions)
        @return bleu
        """
        bp = 1.0 if c > r else math.exp(1-float(r)/c)
        return bp * math.exp(1.0/n * sum(math.log(pn) for pn in smoothing(cc[1:n+1], tc[1:n+1])))
    
    @classmethod
    def no_smoothing(cls, cc, tn):
        """
        Unsmoothed precisions.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        if any(c == 0 for c in cc):
            raise ValueError, 'At least one of the clipped counts is zero: %s' % str(cc)
        for c, t in zip(cc, tn):
            yield float(c) / t
    
    @classmethod
    def p1_smoothing(cls, cc, tn):
        """
        Sum 1 to numerator and denorminator for all orders.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        for c, t in zip(cc, tn):
            yield float(c + 1)/(t + 1)
    
    @classmethod
    def ibm_smoothing(cls, cc, tn):
        """
        IBM smoothing. Assigns a precision of 1/2^k for null counts, where k = 1 for the first n
        whose counts are null.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        k = 0
        for c, t in zip(cc, tn):
            if c > 0:
                yield float(c)/t
            else:
                k += 1
                yield 1.0/math.pow(2, k)

    @classmethod
    def unsmoothed_bleu(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.no_smoothing)
    
    @classmethod
    def bleu_p1(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.p1_smoothing)
    
    @classmethod
    def ibm_bleu(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.ibm_smoothing)
    
    @classmethod
    def get(cls, name):
        if name == 'unsmoothed_bleu':
            return BLEU.unsmoothed_bleu
        if name == 'bleu_p1':
            return BLEU.bleu_p1
        if name == 'ibm_bleu':
            return BLEU.ibm_bleu
        raise Exception, 'Unknown implementation BLEU.%s' % name


