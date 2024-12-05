import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

n_neighbors = 5

# Load dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

data = load_data("outnow.csv")

# Columns and scaling
subjects = data.columns[5:32]
scaler = StandardScaler()

# Sidebar selection
subject_list = subjects
selected_subject = st.sidebar.selectbox("Select a subject:", subject_list)

s_data = data[data[selected_subject] == 1]
# Filter and transform data for PCA
subject_data = data[data[selected_subject] == 1].drop(columns = 'title')
titles = data[data[selected_subject] == 1]['title']
numeric = subject_data.drop(columns=['keywords', 'affiliation_id', 'cited_by_count'])
features_scaled = scaler.fit_transform(numeric)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Add PCA results to the dataset
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]
subject_data["Title"] = titles  # Add titles for hover information

# Create hover text with line breaks
subject_data["hover_text"] = subject_data["Title"].apply(
    lambda x: "<br>".join([x[i:i+80] for i in range(0, len(x), 80)])  # Break lines every 30 characters
)

# Plotly scatter plot
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    title=f"{selected_subject} Data",
    hover_name="hover_text",  # Use the hover_text column directly
)
fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><extra></extra>", customdata=subject_data[["hover_text"]])

# st.plotly_chart(fig, use_container_width=True)

selected_points = plotly_events(fig)

# Nearest neighbors calculation
def get_nearest_neighbors(selected_point, points, k=5):
    tree = KDTree(points)
    _, indices = tree.query(selected_point, k=k)
    return indices

def get_paper(index, df):
    return df.iloc[int(index)]

# Placeholder for click-based interaction
st.write(f'Total sample: {subject_data.shape[0]}')

# Selected point expander
if selected_points:  # Check if any point is selected
    # Get the selected paper
    selected_index = selected_points[0]['pointIndex']
    selected_paper = get_paper(selected_index, s_data)

    # Display selected paper information
    with st.expander(f"Selected paper title: {selected_paper['title']}"):
        cleaned_kw = selected_paper['keywords'].replace(";", ", ")
        st.write(f"Keywords: {cleaned_kw}")

    # Calculate recommendations using nearest neighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute')  # Add 1 to account for the selected paper
    knn.fit(features_scaled)  # Fit the model on scaled features

    # Find neighbors for the selected paper
    selected_point_features = features_scaled[selected_index].reshape(1, -1)
    distances, indices = knn.kneighbors(selected_point_features)

    # Filter out the selected paper from the recommendations
    filtered_indices = [idx for idx in indices[0] if idx != selected_index]
    filtered_distances = [distances[0][i] for i, idx in enumerate(indices[0]) if idx != selected_index]

    # Display recommended papers
    for i, index in enumerate(filtered_indices[:n_neighbors]):  # Limit to `n_neighbors`
        rec_paper = get_paper(index, s_data)
        with st.expander(f"Recommend #{i+1}: {rec_paper['title']}"):
            cleaned_kw = rec_paper['keywords'].replace(";", ", ")
            st.write(f"Keywords: {cleaned_kw}")
            st.write(f"Distance from selected paper: {filtered_distances[i]}")
else:
    st.write("Please select a paper")
