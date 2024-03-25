import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from learning_curve import learning_curve
from preprocess import (
    data_loading,
    data_loading2,
    feature_preparation,
    feature_visualization,
)
from random_forest import RF_classification, grid_search_random_forest
from svm import SVM_classification, grid_search_SVM


def main(data_path):
    # specify the data folder
    """ "Here you need to specify your own path"""
    path = "./ass2/pointclouds-500"

    # conduct feature preparation
    print("Start preparing features")
    feature_preparation(data_path=path)

    # load the data
    print("Start loading data from the local file")
    ID, X, y = data_loading()

    # visualize features
    print("Visualize the features")
    feature_visualization(X=X)

    ID, X, y = data_loading2()

    # SVM classification
    print("Start SVM classification")
    SVM_classification(X, y)

    # RF classification
    print("Start RF classification")
    RF_classification(X, y)

    rf_params = {
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [10, 100, 500, 1000],
        "max_samples": [0.5, 0.75, 1.0],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }

    best_rf_params, best_rf_acc = grid_search_random_forest(
        X, y, param_grid=rf_params
    )

    print("--------------Random Forest--------------")
    print("Best RF accuracy: %5.2f", best_rf_acc)
    print("Best RF parameters:")
    print(best_rf_params)
    print("----------------------------------------")

    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": [
            "linear",
            "poly",
            "rbf",
            "sigmoid",
            # "precomputed",
        ],
        "class_weight": ["balanced", None],
        "max_iter": [-1, 200, 500, 1000],
        "decision_function_shape": ["ovo", "ovr"],
    }

    best_svm_params, best_svm_acc = grid_search_SVM(
        X, y, param_grid=svm_params
    )
    print("--------------SVM--------------")
    # {'n_estimators': 50, 'max_depth': 10, 'criterion': 'gini', 'max_samples': 1.0, 'max_features': 'log2', 'bootstrap': False}
    print("Best SVM accuracy: %5.2f" % best_svm_acc)
    print(
        "{'C': 10, 'kernel': 'linear', 'class_weight': None, 'max_iter': 1000, 'decision_function_shape': 'ovo'}"
    )
    print("Best SVM parameters:")
    print(best_svm_params)
    print("----------------------------------------")
    start = 0.01
    stop = 0.9
    step = 0.05
    num = int((stop - start) / step) + 1
    train_steps = np.linspace(0.01, 0.9, num)

    learning_curve(
        estimator=svm.SVC(
            C=best_svm_params["C"],
            kernel=best_svm_params["kernel"],
            class_weight=best_svm_params["class_weight"],
            max_iter=best_svm_params["max_iter"],
            decision_function_shape=best_svm_params[
                "decision_function_shape"
            ],
        ),
        title="SVM Learning Curve",
        X=X,
        y=y,
        train_steps=train_steps,
    )

    learning_curve(
        estimator=RandomForestClassifier(
            n_estimators=best_rf_params["n_estimators"],
            criterion=best_rf_params["criterion"],
            max_depth=best_rf_params["max_depth"],
            max_samples=(
                None
                if (best_rf_params["bootstrap"] is False)
                else best_rf_params["max_samples"]
            ),
            max_features=best_rf_params["max_features"],
            bootstrap=best_rf_params["bootstrap"],
        ),
        title="RF Learning Curve",
        X=X,
        y=y,
        train_steps=train_steps,
    )


if __name__ == "__main__":
    main("./ass2/pointclouds-500")
