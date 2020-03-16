import sys
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from train_classifier import load_data, tokenize


def tune_classifier(classifier, X, Y, category_names, params_grid):
    """Performs an exhaustive search over the space of parameter values defined in params_grid, looking
    for the combination that gives the best results training and testing the classifier on the dataset
    :classifier: The classification model to be tunned
    :X: Independent variables of the dataset
    :Y: Dependent variables of the dataset (categories)
    :params_grid: Dictionary defining the values of each parameter that should be tried in the tunning
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    classifier.fit(X_train, Y_train)

    cv = GridSearchCV(classifier, param_grid=params_grid, cv=5, n_jobs=-1)

    print("Searching for best parameters...")

    cv.fit(X_train, Y_train)

    print(f"- BEST PARAMETERS: \n{cv.best_params_}\n")

    Y_pred = cv.predict(X_test)

    # Compute f-score, recall and precision of each class
    for j, column in enumerate(category_names):
        cur_class_report = classification_report(Y_test[:, j], Y_pred[:, j], output_dict=False)
        print(f"Classification report for {column}: \n\n{cur_class_report}\n")

    # Compute mean accuracy over all categories
    print(f"* MEAN ACCURACY: {cv.score(X_test, Y_test)}\n")

    return cv.best_params_


def get_params_grid_for_random_forest():
    """
    Convenience function that returns the grid of parameters to be explored by GridSearchCV when using
    Random Forest as estimator
    :return: Grid of parameters for a classifier pipele, with a CountVectorizer and TfIdfTransformer
    """
    return {
        "vctzr__tokenizer": [tokenize, None],
        "tfidf__smooth_idf": [True, False],
        "tfidf__use_idf": [True, False],
        "tfidf__sublinear_tf": [True, False],
        "clsfr__estimator__criterion": ["gini", "entropy"],
        "clsfr__estimator__max_depth": [10, 30, 50],
        "clsfr__estimator__min_samples_split": [2, 4, 5],
        "clsfr__estimator__n_estimators": [50, 75, 100],
        "clsfr__estimator__warm_start": [False, True]
    }


def get_params_grid_for_svc():
    """
    Convenience function that returns the grid of parameters to be explored by GridSearchCV when using
    Support Vector Classifier (SVC) as estimator
    :return: Grid of parameters for a classifier pipele, with a CountVectorizer and TfIdfTransformer
    """
    return {
        "vctzr__tokenizer": [tokenize, None],
        "tfidf__smooth_idf": [True],
        "tfidf__use_idf": [True],
        "tfidf__sublinear_tf": [True, False],
        "clsfr__estimator__kernel": ["poly", "rbf"],
        "clsfr__estimator__C": [0.5, 1.0],
        "clsfr__estimator__probability": [True, False],
        "clsfr__estimator__gamma": ["scale"],
        "clsfr__estimator__class_weight": [None, "balanced"],
        "clsfr__estimator__decision_function_shape": ["ovo", "ovr"]
    }


def get_params_grid_for_linear_svc():
    """
    Convenience function that returns the grid of parameters to be explored by GridSearchCV when using
    Linear Support Vector Classifier (LinearSVC) as estimator
    :return: Grid of parameters for the classifier pipeline, with a CountVectorizer and TfIdfTransformer
    """
    return {
        "vctzr__tokenizer": [tokenize, None],
        "tfidf__smooth_idf": [True, False],
        "tfidf__use_idf": [True, False],
        "tfidf__sublinear_tf": [True, False],
        "clsfr__estimator__class_weight": [None, "balanced"],
        "clsfr__estimator__fit_intercept": [True, False]
    }


def main():
    if len(sys.argv):
        db_path = sys.argv[1]

        # Classifier to predict categories based on the vectorized text
        svc_clsfr = SVC()

        # We have a multiple classifications per input text, that is, we have a two-dimensional target variable.
        # Use the MultiOutputClassifier to train a different Classifier per target variable
        clsfr = Pipeline([
            ("vctzr", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer(use_idf=True, smooth_idf=True)),
            ("clsfr", MultiOutputClassifier(svc_clsfr))
        ])

        parameters = get_params_grid_for_svc()

        X, Y, categories = load_data(db_path)

        tune_classifier(clsfr, X, Y, categories, parameters)

    else:
        print("Please provide the filepath of the disaster messages database as the first argument")


if __name__ == "__main__":
    main()
