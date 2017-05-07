import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData, verbose=False):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    word_ids = sorted(test_set.get_all_Xlengths().keys())

    for id in word_ids:
        X, lengths = test_set.get_item_Xlengths(id)
        scores = dict()
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except Exception as e:
                if verbose:
                    print("{}: Exception '{}'".format(word, e))
                    print("Set logL to -Inf")
                logL = float('-Inf')
            scores[word] = logL
        probabilities.append(scores)
        guesses.append(max(scores, key=scores.get))

    return probabilities, guesses
