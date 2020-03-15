from train_classifier import load_data, tokenize


def tune_classifier(classifier, X, Y, params_grid):
    """Performs an exhaustive search over the space of parameter values defined in params_grid, looking
    for the combination that gives the best results training and testing the classifier on the dataset
    :classifier: The classification model to be tunned
    :X: Independent variables of the dataset
    :Y: Dependent variables of the dataset (categories)
    :params_grid: Dictionary defining the values of each parameter that should be tried in the tunning
    """
    pass


if __name__ == "__main__":

    parameters = {
        "vctzr__tokenizer": [tokenize, None],
        "tfidf__smooth_idf": [True],
        "tfidf__use_idf": [True],
        # "tfidf__sublinear_tf": [True, False],
        "clsfr__estimator__criterion": ["gini", "entropy"],
        "clsfr__estimator__max_depth": [30, 50],
        # "clsfr__estimator__min_samples_split": [2, 4, 5],
        "clsfr__estimator__n_estimators": [75, 100],
        # "clsfr__estimator__warm_start": [False, True]
    }

