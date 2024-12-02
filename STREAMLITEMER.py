import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

data = load_data("out.csv")
data2 = load_data("out2.csv")

# Columns and scaling
subjects = data.columns[1:28]
scaler = StandardScaler()

# Sidebar selection
subject_list = subjects
selected_subject = st.sidebar.selectbox("Select a subject:", subject_list)

# Filter and transform data for PCA
subject_data = data[data[selected_subject] == 1]
titles = data2[data[selected_subject] == 1]['title']
features_scaled = scaler.fit_transform(subject_data)  # Assuming features start at column 28
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Add PCA results to the dataset
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]

# Plotly scatter plot
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    hover_data={"Research Name": titles, "PCA1": False, "PCA2": False},  # Show only Research Name on hover
    title=f"{selected_subject} Data",
)
st.plotly_chart(fig, use_container_width=True)

# Nearest neighbors calculation
def get_nearest_neighbors(selected_point, points, k=5):
    tree = KDTree(points)
    _, indices = tree.query(selected_point, k=k)
    return indices

# Placeholder for click-based interaction
st.write("Click functionality currently requires enhancements through JavaScript.")
st.write("Use hover to see the research point name.")
