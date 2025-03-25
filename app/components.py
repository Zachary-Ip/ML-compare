import streamlit as st
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

from sklearn.metrics import r2_score


def dataset_selector():
    st.subheader("🔬 Choose a Machine Learning Task")
    task = st.selectbox(
        "Select Task", ["Regression", "Classification", "Clustering"], key="task"
    )

    dataset_options = {
        "Regression": ["Diabetes", "California Housing"],
        "Classification": ["Iris", "Wine"],
        "Clustering": ["Iris", "Wine", "Diabetes", "California Housing"],
    }

    dataset_name = st.selectbox(
        "Select a Dataset", dataset_options[task], key="dataset"
    )
    data, target, feature_names = utils.load_dataset(dataset_name)
    return task, data, target, feature_names


def choose_features(features):
    st.subheader("Preview features to train on")

    f1 = st.selectbox("Feature 1", features, key="Feature 1", index=0)
    f2 = st.selectbox("Feature 2", features, key="Feature 2", index=1)

    return f1, f2


def plot_features(feats, data, f1, f2):

    st.write(f"Data shape {data.shape}")
    f1_data = data[:, feats.index(f1)]
    f2_data = data[:, feats.index(f2)]

    g = sns.jointplot(x=f1_data, y=f2_data)
    sns.scatterplot(x=f1_data, y=f2_data, s=5, color=".15", ax=g.ax_joint)

    g.ax_joint.set_xlabel(f1)
    g.ax_joint.set_ylabel(f2)
    st.pyplot(g)
    select_features = np.column_stack((f1_data, f2_data))
    return select_features


def model_selector(task, key):
    model_options = {
        "Regression": ["None", "Linear", "AdaBoostRegressor"],
        "Classification": ["None", "Logistic", "Random Forest"],
        "Clustering": ["None", "K-Means", "Hierarchical", "DBScan"],
    }
    st.subheader("Select an ML algorithm")
    return st.selectbox("Select a method", model_options[task], key=key, index=0)


def train_model(task, method, X, y):
    method_details = {
        "None": {
            "fn": None,
            "params": None,
        },
        "Linear": {"fn": sklearn.linear_model.LinearRegression, "params": None},
        "AdaBoostRegressor": {
            "fn": sklearn.ensemble.AdaBoostRegressor,
            "params": {
                "n_estimators": {"type": "int", "min": 1, "max": 100, "value": 50},
                "learning_rate": {
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "value": 1.0,
                },
                "loss": {
                    "type": "select",
                    "options": ["linear", "square", "exponential"],
                    "value": "linear",
                },
            },
        },
        "Logistic": {
            "fn": sklearn.linear_model.LogisticRegression,
            "params": {
                "solver": {
                    "type": None,
                    "value": "saga",
                },
                "penalty": {
                    "type": "select",
                    "options": [None, "l1", "l2", "elasticnet"],
                    "value": "l1",
                },
                "C": {"type": "float", "min": 0.0, "max": 10.0, "value": 1.0},
            },
        },
        "Random Forest": {
            "fn": sklearn.ensemble.RandomForestClassifier,
            "params": {
                "n_estimators": {"type": "int", "min": 1, "max": 200, "value": 100},
                "criterion": {
                    "type": "select",
                    "options": ["gini", "entropy", "log_loss"],
                    "value": "gini",
                },
            },
        },
        "K-Means": {
            "fn": sklearn.cluster.KMeans,
            "params": {
                "n_clusters": {"type": "int", "min": 1, "max": 10, "value": 2},
                "init": {
                    "type": "select",
                    "options": ["k-means++", "random"],
                    "value": "k-means++",
                },
            },
        },
        "Hierarchical": {
            "fn": sklearn.cluster.AgglomerativeClustering,
            "params": {
                "n_clusters": {
                    "type": None,
                    "value": None,
                },
                "metric": {
                    "type": "select",
                    "options": ["euclidean", "manhattan", "log_loss"],
                    "value": "euclidean",
                },
                "distance_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1000.0,
                    "value": 1.0,
                },
            },
        },
        "DBScan": {
            "fn": sklearn.cluster.DBSCAN,
            "params": {
                "eps": {"type": "float", "min": 0.0, "max": 10.0, "value": 0.5},
                "min_samples": {"type": "int", "min": 1, "max": 25, "value": 5},
            },
        },
    }

    details = method_details[method]
    # Choose hyperparameters
    if details["params"]:
        with st.sidebar:
            for key, value in details["params"].items():
                if value["type"] == "int":
                    val = st.slider(
                        key,
                        step=1,
                        value=value["value"],
                        min_value=value["min"],
                        max_value=value["max"],
                    )
                elif value["type"] == "float":
                    val = st.slider(
                        key,
                        value=value["value"],
                        min_value=value["min"],
                        max_value=value["max"],
                    )
                elif value["type"] == "select":
                    val = st.segmented_control(
                        key, value["options"], default=value["value"]
                    )
                else:
                    val = value["value"]
                details["params"][key]["value"] = val
        # Extract parameter values
        params = {key: value["value"] for key, value in details["params"].items()}
    else:
        params = None

    # Train model
    fn = details["fn"]

    # Call the function with parameters
    if method == "hierarchical":
        fn_instance = fn(n_clusters=None, **params)
    if params and fn:
        fn_instance = fn(**params)
    elif fn:
        fn_instance = fn()
    else:
        st.stop()

    if task != "Clustering":
        # For regression and classification, we need to split the data into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        fn_instance.fit(X_train, y_train)

        labels = fn_instance.predict(X_test)
    else:
        fn_instance.fit(X, y)
        labels = fn_instance.fit_predict(X)

    # Unique cluster labels
    unique_labels = set(labels)

    if task == "Clustering":
        # Define colors (black for noise)
        colors = plt.cm.cool(np.linspace(0, 1, len(unique_labels)))
        cluster_colors = {label: color for label, color in zip(unique_labels, colors)}

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for label in unique_labels:
            # Select points of the current cluster
            mask = labels == label
            color = "black" if label == -1 else cluster_colors[label]
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[color],
                label=f"Cluster {label}" if label != -1 else "Noise",
                edgecolors="k",
            )

        plt.title(method)
        plt.legend()
        st.pyplot(plt)

    elif task == "Regression":
        plot_regression_results(X_test, y_test, labels, method)
    elif task == "Classification":
        plot_classification_results(fn_instance, X_test, y_test, labels)


def plot_regression_results(X_test, y_test, y_pred, method):
    """
    Plots regression model predictions against actual values.
    Handles both single-variable and multi-variable regression.
    """
    plt.figure(figsize=(8, 6))

    # Single-variable regression (2D plot)
    if X_test.shape[1] == 1:
        plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)
        plt.scatter(X_test, y_pred, color="red", marker="x", label="Predicted Data")
        plt.plot(
            np.sort(X_test, axis=0),
            np.sort(y_pred, axis=0),
            color="black",
            linestyle="--",
        )
        plt.xlabel("Feature")
        plt.ylabel("Target")

    # Multi-variable regression (y_pred vs y_test scatter)
    else:
        plt.scatter(y_test, y_pred, color="purple", alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="black",
            linestyle="--",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")

    plt.title(f"{method} Model Performance (R² = {r2_score(y_test, y_pred):.2f})")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def plot_classification_results(model, X_test, y_test, y_pred):
    """
    Plots classification model results using:
    - Confusion Matrix
    - Decision Boundary (only for 2D data)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")
    ax[0].set_title("Confusion Matrix")

    # Decision Boundary (only for 2D data)
    if X_test.shape[1] == 2:
        plot_decision_regions(X_test, y_test, clf=model, ax=ax[1])
        ax[1].set_title("Decision Boundary")

    plt.tight_layout()
    st.pyplot(fig)

    # Print classification report
    st.write("Classification Report:\n", classification_report(y_test, y_pred))
