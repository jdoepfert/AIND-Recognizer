import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    According to https://en.wikipedia.org/wiki/Hidden_Markov_model, there are
    n*(n-1) transition parameters, where n is the number of states.
    Furthermore, for reach of the n states, 2*m parameters controlling the
    multivariate Gaussian (assuming a diagonal covariance matrix as used here),
    where m is the number of dimensions (i.e. the number of features).
    Therefore, the number of parameters p can be computed as follows:
    p = n*(n-1) + n*(2*m)
    """

    def select(self, verbose=False):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components


        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        possible_n_components = list(range(self.min_n_components,
                                           self.max_n_components+1))
        scores = []
        for n_components in possible_n_components:
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                n = model.n_components  # equals `n_components`
                m = model.n_features
                p = n*(n-1) + n*(2*m)  # see class docstring
                N = len(self.X)  # each feature vector in `X` is a data point
                logN = np.log(N)
                BIC = -2 * logL + p * logN
            except Exception as e:
                if verbose:
                    print("n = {}: Exception '{}'".format(n_components, e))
                    print("Set BIC to Inf")
                BIC = float('Inf')
            scores.append(BIC)

        best_num_components = possible_n_components[scores.index(min(scores))]
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    M: number of classes (i.e. words)
    log(P(X(i)): log likelihood of word i
    '''

    def select(self, verbose=False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        possible_n_components = list(range(self.min_n_components,
                                           self.max_n_components+1))
        M = len(self.hwords.keys())  # number of classes/words
        scores = []
        for n_components in possible_n_components:
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                other_words = [w for w in self.hwords.keys() if self.this_word != w]
                sum_anti_likelihoods = 0
                for word in other_words:
                    X, lengths = self.hwords[word]
                    sum_anti_likelihoods += model.score(X, lengths)
                DIC = logL - 1/(M - 1) * sum_anti_likelihoods
            except Exception as e:
                if verbose:
                    print("n = {}: Exception '{}'".format(n_components, e))
                    print("Set DIC to -Inf")
                DIC = float('-Inf')
            scores.append(DIC)

        best_num_components = possible_n_components[scores.index(max(scores))]
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self, verbose=False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold(2)
        possible_n_components = list(range(self.min_n_components,
                                           self.max_n_components+1))
        scores = []
        for n_components in possible_n_components:
            cv_score = 0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    # training
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    model = self.base_model(n_components)

                    # testing
                    X, lengths = combine_sequences(cv_test_idx, self.sequences)
                    cv_score += model.score(X, lengths)
            except Exception as e:
                if verbose:
                    print("n = {}: Exception '{}'".format(n_components, e))
                    print("Set cv score to -Inf")
                    cv_score = -float('Inf')
            scores.append(cv_score)

        best_num_components = possible_n_components[scores.index(max(scores))]
        # reset self.X and self.lenghts to original value
        self.X, self.lengths = self.hwords[self.this_word]
        # return fitted model with best parameters
        return self.base_model(best_num_components)
