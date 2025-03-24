import streamlit as st
import utils

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
def dataset_selector():
    st.subheader("🔬 Choose a Machine Learning Task")
    task = st.selectbox("Select Task", ["Regression", "Classification", "Clustering"], key="task")
    
    dataset_options = {
        "Regression": ["Diabetes", "California Housing"],
        "Classification": ["Iris", "Wine"],
        "Clustering": ["Iris", "Wine", "Diabetes", "California Housing"]
    }
    
    dataset_name = st.selectbox("Select a Dataset", dataset_options[task], key="dataset")
    data, target, feature_names = utils.load_dataset(dataset_name)
    return task, data, target, feature_names


def choose_features(features):
    st.subheader("Preview features to train on")

    f1 = st.selectbox("Feature 1", features, key="Feature 1", index=0)
    f2 = st.selectbox("Feature 2", features, key="Feature 2", index=1)

    return f1, f2

def plot_features(feats, data, f1, f2):

    st.write(f"Data shape {data.shape}")
    f1_data  = data[:,feats.index(f1)]
    f2_data = data[:,feats.index(f2)]

    g = sns.jointplot(x=f1_data, y=f2_data)
    sns.scatterplot(x=f1_data, y=f2_data, s=5, color=".15", ax=g.ax_joint)

    g.ax_joint.set_xlabel(f1)
    g.ax_joint.set_ylabel(f2)
    st.pyplot(g)

def model_selector(task, key):
    model_options = {
        "Regression": ["None", "Linear", "Ridge", "Lasso", "Elastic Net", "AdaBoostRegressor", "Extr"],
        "Classification": ["None", "Logistic", "Naive Bayes", "Random Forest", "AdaBoost", "Gradient Boost", "XGBoost"],
        "Clustering": ["None", "K-Means", "Hierarchical", "DBScan"]
    }
    st.subheader("Select an ML algorithm")
    return st.selectbox("Select a method", model_options[task], key=key, index=0)


def train_model(method):
    name2function = {
        "None":{
            "fn": None,
            "params": None,
        },
        "Linear":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": None
        }, 
        "Ridge":{
            "fn": sklearn.linear_model.Ridge,
            "params": {
                "alpha": {
                    "type":"float",
                    "min": 0,
                    "value": 1
                }
            }
        },
        "Lasso":{
            "fn": sklearn.linear_model.Lasso,
            "params": {
                "alpha": {
                    "type":"float",
                    "min": 0,
                    "value": 1
                }
            }
        },
        "Elastic Net":{
            "fn": sklearn.linear_model.ElasticNet,
            "params": {
                "alpha": {
                    "type":"float",
                    "min": 0,
                    "value": 1
                },
                "l1_ratio": {
                    "type":"float",
                    "min":0,
                    "max":1,
                    "value": 0.5
                }
            }
        },
        "AdaBoostRegressor": {
            "fn": sklearn.ensemble.AdaBoostRegressor,
            "params": {
                "n_estimators": {
                    "type":"int",
                    "min": 1,
                    "value": 50
                },
                "learning_rate": {
                    "type":"float",
                    "min": 0,
                    "value": 1
                },
                "loss":{
                    "type":"select",
                    "options": ["linear", "square", "exponential"],
                    "value": "linear"
                }
            }
        },
        # "Naive Bayes":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "Random Forest":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "AdaBoost":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "Gradient Boost":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "XGBoost":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "K-Means":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "Hierarchical":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
        # "DBScan":{
        #     "fn": sklearn.linear_model.LinearRegression,
        #     "params": []
        # },
    }
    method_info = name2function[method]
    st.write(method_info)
    # Choose hyperparameters
    if method_info["params"]:
        with st.sidebar:
            for key, value in method_info["params"].items():
                if value["type"] == "int":
                    val = st.number_input(key, step=1)
                elif value == "float":
                    val = st.number_input(key)
                elif isinstance(value, list):
                    val = st.selectbox(key,value)
                    val = st.number_input(key, step=1)
                method_info["params"][key]["value"] = val
    




    

