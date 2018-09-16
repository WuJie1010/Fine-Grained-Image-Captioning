#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
from collections import defaultdict
import numpy as np
import pdb
import math
import pickle
import os

def precook_test(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    counts_token = []
    for k in xrange(1,n+1): #1,2,3,4
        if k==1:
            for i in xrange(len(words)):
                token_ngram = tuple(words[i:i + 1])
                counts_token.append(token_ngram)
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts, counts_token

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1,n+1): #1,2,3,4
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    counts, counts_token = precook_test(test, n, True)
    return counts, counts_token

class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, df_mode="corpus", test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.test_token = []
        self.df_mode = df_mode
        self.ref_len = None
        if self.df_mode != "corpus":
            pkl_file = pickle.load(open(os.path.join('data', df_mode + '.p'),'r'))
            self.ref_len = pkl_file['ref_len']
            self.document_frequency = pkl_file['document_frequency']
        self.cook_append(test, refs)
    
    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                counts, counts_token = cook_test(test)
                self.ctest.append(counts) ## N.B.: -1
                self.test_token.append(counts_token)
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match
                self.test_token.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self, loader):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            vec_log = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            norm_log = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.iteritems():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))

                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                vec_log[n][ngram] = float(term_freq)*(np.log(self.ref_len) - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)
                norm_log[n] += pow(vec_log[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            norm_log = [np.sqrt(n) for n in norm_log]
            return vec, norm, vec_log, norm_log, length

        def sim(test_token, vec_hyp, vec_log_hyp, vec_ref, vec_log_ref, norm_hyp, norm_log_hyp, norm_ref, norm_log_ref, length_hyp, length_ref):

            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val_log = np.array([0.0 for _ in range(self.n)])
            val = np.array([0.0 for _ in range(self.n)])
            token_score = np.array([0.0 for _ in range(len(test_token))])

            threshold1 = 1.0
            threshold_all = 5.0

            for i, token in enumerate(test_token):

                token_val = np.array([0.0 for _ in range(self.n)])
                for n in range(self.n):
                    for (ngram, count) in vec_log_hyp[n].iteritems():
                        #print(ngram, vec_log_hyp[n][ngram])
                        if token[0] in ngram:

                            if vec_log_hyp[0][token] > threshold1 and vec_log_hyp[n][ngram] >= threshold_all:
                                #print(ngram, vec_log_hyp[n][ngram])
                                #print(loader.get_vocab()[token[0]])
                                """
                                txt=[]
                                for i in range(len(ngram)):
                                    if list(ngram)[i] != str(0):
                                        txt.append(loader.get_vocab()[list(ngram)[i]])
                                if vec_log_ref[n][ngram] >0:
                                    print(txt, vec_log_hyp[n][ngram])
                                """
                                token_val[n] += min(vec_log_hyp[n][ngram], vec_log_ref[n][ngram]) * vec_log_ref[n][ngram]

                    if (norm_log_hyp[n] != 0) and (norm_log_ref[n] != 0):
                        token_val[n] /= (norm_log_hyp[n]*norm_log_ref[n])

                    assert(not math.isnan(token_val[n]))
                    token_val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))

                token_score[i]=np.sum(token_val)
            #print(token_score)
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].iteritems():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))

            for n in range(self.n):
                for (ngram, count) in vec_log_hyp[n].iteritems():
                    val_log[n] += min(vec_log_hyp[n][ngram], vec_log_ref[n][ngram]) * vec_log_ref[n][ngram]

                if (norm_log_hyp[n] != 0) and (norm_log_ref[n] != 0):
                    val_log[n] /= (norm_log_hyp[n]*norm_log_ref[n])

                assert(not math.isnan(val_log[n]))
                # vrama91: added a length based gaussian penalty
                val_log[n] *= np.e**(-(delta**2)/(2*self.sigma**2))

            return val, val_log, token_score

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        log_scores = []
        token_scores = []
        for test, test_token, refs in zip(self.ctest, self.test_token, self.crefs):
            # compute vector for test captions
            vec, norm, vec_log, norm_log, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            log_score = np.array([0.0 for _ in range(self.n)])
            token_score = np.array([0.0 for _ in range(len(test_token))])
            for ref in refs:
                vec_ref, norm_ref, vec_log_ref, norm_log_ref, length_ref = counts2vec(ref)
                val, val_log, token= sim(test_token, vec, vec_log, vec_ref, vec_log_ref, norm, norm_log, norm_ref, norm_log_ref, length, length_ref)
                score += val
                log_score += val_log
                token_score += token

            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)

            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)

            log_score_avg = np.mean(log_score)
            # divide by number of references
            log_score_avg /= len(refs)
            # multiply score by 10
            log_score_avg *= 10.0
            # append score of an image to the score list
            log_scores.append(log_score_avg)


            # change by vrama91 - mean of ngram scores, instead of sum
            # divide by number of references
            token_score /= len(refs)
            # multiply score by 10
            token_score *= 10.0
            #token_score *= len(test_token)
            # append score of an image to the score list
            token_scores.append(token_score)
            #print(token_score)
        return scores, log_scores, token_scores

    def compute_score(self, loader, option=None, verbose=0):
        # compute idf
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            # assert to check document frequency
            assert(len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        # compute cider score
        score, log_score, token_score = self.compute_cider(loader)
        #print(np.mean(np.array(score)))
        return np.array(score), np.array(log_score), np.array(token_score)
