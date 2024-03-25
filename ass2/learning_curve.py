from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def learning_curve(
    estimator,
    title,
    X,
    y,
    train_steps=np.linspace(0.1, 0.9, 9),
):
    train_errors = []
    test_errors = []
    train_sizes = []

    for train_size_ration in train_steps:
        train_size = int(len(X) * train_size_ration)
        train_sizes.append(train_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=20, stratify=y
        )
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        test_pred = estimator.predict(X_test)

        train_error = 1.0 - accuracy_score(y_train, train_pred)
        test_error = 1.0 - accuracy_score(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)
        print("accuracy score: ", accuracy_score(y_train, train_pred))
    print("train error: ", train_errors)
    print("test error: ", test_errors)

    plt.figure()
    plt.plot(
        train_sizes,
        train_errors,
        label="Training errors",
        color="blue",
    )
    plt.plot(
        train_sizes,
        test_errors,
        label="Test errors",
        color="green",
    )
    plt.title(title if title is not None else "Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
