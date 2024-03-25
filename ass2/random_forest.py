import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold, train_test_split


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=20, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("RF accuracy: %5.2f" % acc)
    conf = confusion_matrix(y_test, y_preds)
    print("confusion matrix", conf)
    class_report = classification_report(y_test, y_preds, output_dict=True)

    importance = clf.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in clf.estimators_], axis=0
    )
    indices = np.argsort(importance)[::-1]
    for f in range(X.shape[1]):
        print(
            "%d. feature %d (%f)"
            % (f + 1, indices[f], importance[indices[f]])
        )
    return (acc, conf, class_report)


def grid_search_random_forest(X, y, param_grid=None, test_size=0.4):

    best_params = None
    highest_acc = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=20, stratify=y
    )

    print(
        "total iter: ",
        len(param_grid["n_estimators"])
        * len(param_grid["max_depth"])
        * len(param_grid["criterion"])
        * len(param_grid["max_samples"])
        * len(param_grid["max_features"])
        * len(param_grid["bootstrap"]),
    )

    i = 0
    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            for criterion in param_grid["criterion"]:
                for max_samples in param_grid["max_samples"]:
                    for max_features in param_grid["max_features"]:
                        for bootstrap in param_grid["bootstrap"]:

                            clf = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                criterion=criterion,
                                max_samples=(
                                    None
                                    if (bootstrap is False)
                                    else max_samples
                                ),
                                max_features=max_features,
                                bootstrap=(bootstrap if bootstrap else False),
                                random_state=20,
                            )
                            clf.fit(X_train, y_train)
                            y_preds = clf.predict(X_test)
                            acc = accuracy_score(y_test, y_preds)
                        if acc > highest_acc:
                            highest_acc = acc
                            best_params = {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "criterion": criterion,
                                "max_samples": max_samples,
                                "max_features": max_features,
                                "bootstrap": (
                                    bootstrap if bootstrap else False
                                ),
                            }
    return (best_params, highest_acc)
