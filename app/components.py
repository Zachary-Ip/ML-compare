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
        "Regression": ["None", "Linear", "Ridge", "Lasso", "Elastic Net", "Polynomial", "Naive Bayes", "AdaBoostRegressor", "Extr"],
        "Classification": ["None", "Logistic", "Naive Bayes", "Random Forest", "AdaBoost", "Gradient Boost", "XGBoost"],
        "Clustering": ["None", "K-Means", "Hierarchical", "DBScan"]
    }
    st.subheader("Select an ML algorithm")
    return st.selectbox("Select a method", model_options[task], key=key, index=0)


def train_model(method):
    name2function = {
        "Linear":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        }, 
        "Ridge":{
            "fn": sklearn.linear_model.Ridge,
            "params": []
        },
        "Lasso":{
            "fn": sklearn.linear_model.Lasso,
            "params": []
        },
        "Elastic Net":{
            "fn": sklearn.linear_model.ElasticNet,
            "params": []
        },
        "Polynomial":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Naive Bayes":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Logistic": {
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Naive Bayes":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Random Forest":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "AdaBoost":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Gradient Boost":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "XGBoost":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "K-Means":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "Hierarchical":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
        "DBScan":{
            "fn": sklearn.linear_model.LinearRegression,
            "params": []
        },
    }

    

