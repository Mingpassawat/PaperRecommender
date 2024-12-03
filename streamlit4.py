import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load datasets
@st.cache_data
def load_data():
    return pd.read_csv("data.csv"), pd.read_csv("data copy.csv")

data, data_copy = load_data()

# Columns and scaling
subjects = data.columns[1:28]
scaler = StandardScaler()

# Sidebar selection
subject_list = subjects
selected_subject = st.sidebar.selectbox("Select a subject:", subject_list)

# Filter and transform data for PCA
subject_data = data[data[selected_subject] == 1]
features_scaled = scaler.fit_transform(subject_data.iloc[:, 28:])  # Assuming features start at column 28
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Add PCA results to the dataset
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]

# Interactive scatter plot with hover
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    hover_data={"File": True, "PCA1": False, "PCA2": False},  # Show only File on hover
    title=f"{selected_subject} Data",
)
st.plotly_chart(fig, use_container_width=True)

# Nearest neighbors calculation
def get_nearest_neighbors(selected_point, points, k=5):
    tree = KDTree(points)
    _, indices = tree.query(selected_point, k=k)
    return indices

# Placeholder for selected point
clicked_point = st.session_state.get("clicked_point", None)

if clicked_point:
    # Find the selected point
    selected_point = np.array([clicked_point["x"], clicked_point["y"]])
    points = subject_data[["PCA1", "PCA2"]].values
    indices = get_nearest_neighbors(selected_point, points)

    # Get neighbor filenames
    neighbors = subject_data.iloc[indices]
    neighbor_files = neighbors["File"].values

    # Match titles using the 'filename' column in data copy.csv
    neighbor_titles = data_copy[data_copy["filename"].isin(neighbor_files)]["title"]

    # Display results
    st.write("Nearest Neighbors' Titles:")
    st.write(neighbor_titles)

