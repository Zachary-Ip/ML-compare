from sklearn import datasets


def load_dataset(name):
    """
    Load Data, Targets, and features from sklearn datasets
    """
    if name == "Iris":  # classification
        dataset = datasets.load_iris()
    elif name == "Wine":  # Classification
        dataset = datasets.load_wine()
    elif name == "Diabetes":  # regression
        dataset = datasets.load_diabetes()
    elif name == "California Housing":  # regression
        dataset = datasets.fetch_california_housing()
    else:
        return None
    return dataset.data, dataset.target, dataset.feature_names
