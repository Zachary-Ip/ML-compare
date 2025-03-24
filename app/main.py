import streamlit as st
import components

st.set_page_config(page_title="Home", layout="wide")

st.title("🔍 Machine Learning Visualizer")
st.write("Explore different ML models interactively.")

with st.sidebar:
    task, data, target, feats = components.dataset_selector()
    f1, f2 = components.choose_features(feats)

st.write("Plotting features")
components.plot_features(feats, data, f1, f2)

with st.sidebar:
    m1 = components.model_selector(task, "method_1")
    m2 = components.model_selector(task, "method_2")

if m2 != "None":
    col1, col2 = st.columns(2)
        

     