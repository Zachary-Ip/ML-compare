from sklearn import datasets

# --- Load Dataset ---
def load_dataset(name):
    if name == "Iris":
        dataset = datasets.load_iris()
    elif name == "Wine":
        dataset = datasets.load_wine()
    elif name == "Digits":
        dataset = datasets.load_digits()
    elif name == "Diabetes":
        dataset = datasets.load_diabetes()
    elif name == "Boston Housing":
        dataset = datasets.load_boston()
    else:
        return None
    return dataset.data, dataset.target, dataset.feature_names