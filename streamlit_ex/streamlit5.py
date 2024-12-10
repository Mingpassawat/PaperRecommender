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

# Debugging to ensure data integrity
st.write("Subject Data Preview:", subject_data.head())
st.write("Columns in Subject Data:", subject_data.columns)

if subject_data.empty:
    st.error("No data available for the selected subject.")
else:
    # Merge titles from data2 based on filename (adjust columns if necessary)
    merged_data = subject_data.merge(data2, left_on="File", right_on="filename", how="left")

    # Extract features for PCA (adjust column range based on dataset structure)
    feature_columns = merged_data.iloc[:, 28:]  # Assuming features start at column 28
    if feature_columns.empty:
        st.error("No numeric features found for PCA.")
    else:
        features_scaled = scaler.fit_transform(feature_columns)

        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)

        # Add PCA results to the dataset
        merged_data["PCA1"] = features_2d[:, 0]
        merged_data["PCA2"] = features_2d[:, 1]

        # Plotly scatter plot with hover functionality for titles
        fig = px.scatter(
            merged_data,
            x="PCA1",
            y="PCA2",
            hover_data={"title": True, "PCA1": False, "PCA2": False},  # Show research titles on hover
            title=f"{selected_subject} Data",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Nearest neighbors calculation
        def get_nearest_neighbors(selected_point, points, k=5):
            tree = KDTree(points)
            _, indices = tree.query(selected_point, k=k)
            return indices

        # Click-based interaction (currently a placeholder)
        st.write("Click functionality currently requires enhancements through JavaScript.")
        st.write("Use hover to see the research titles.")
