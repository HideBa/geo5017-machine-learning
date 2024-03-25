from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def SVM_classification(X, y, kernel="rbf"):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=20, stratify=y
    )
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)
    class_report = classification_report(y_test, y_preds, output_dict=True)
    print("report: ", class_report)

    return (acc, conf, class_report)


def grid_search_SVM(X, y, param_grid=None, test_size=0.4):
    best_params = None
    highest_acc = 0

    print(
        "total iter: ",
        len(param_grid["C"])
        * len(param_grid["kernel"])
        * len(param_grid["class_weight"])
        * len(param_grid["max_iter"])
        * len(param_grid["decision_function_shape"]),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=20, stratify=y
    )

    i = 0
    for C in param_grid["C"]:
        for decision_function_shape in param_grid["decision_function_shape"]:
            for kernel in param_grid["kernel"]:
                for class_weight in param_grid["class_weight"]:
                    for max_iter in param_grid["max_iter"]:
                        i += 1

                        clf = svm.SVC(
                            C=C,
                            decision_function_shape=decision_function_shape,
                            kernel=kernel,
                            class_weight=class_weight,
                            max_iter=max_iter,
                        )
                        clf.fit(X_train, y_train)
                        y_preds = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_preds)
                        if acc > highest_acc:
                            highest_acc = acc
                            best_params = {
                                "C": C,
                                "kernel": kernel,
                                "class_weight": class_weight,
                                "max_iter": max_iter,
                                "decision_function_shape": decision_function_shape,
                            }
    return (best_params, highest_acc)
