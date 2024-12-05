import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events

# Load dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

data = load_data("outnow.csv")
data2 = load_data("out2.csv")

# Columns and scaling
subjects = data.columns[5:32]
scaler = StandardScaler()

# Sidebar selection
subject_list = subjects
selected_subject = st.sidebar.selectbox("Select a subject:", subject_list)

# Filter and transform data for PCA
subject_data = data[data[selected_subject] == 1].drop(columns = 'title')
titles = data[data[selected_subject] == 1]['title']
numeric = subject_data.drop(columns=['keywords', 'affiliation_id', 'cited_by_count'])
features_scaled = scaler.fit_transform(numeric)  # Assuming features start at column 28
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Add PCA results to the dataset
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]
subject_data["Title"] = titles  # Add titles for hover information

# Create hover text with line breaks
subject_data["hover_text"] = subject_data["Title"].apply(
    lambda x: "<br>".join([x[i:i+35] for i in range(0, len(x), 35)])  # Break lines every 30 characters
)

# # Create a hover text column with full title
# subject_data["hover_text"] = subject_data["Title"].apply(
#     lambda x: f"Research Name: {x}".format(0, 0)
# )

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


# Placeholder for click-based interaction
st.write(f'Total sample: {subject_data.shape[0]}')
st.write(selected_points)
st.write("Click functionality currently requires enhancements through JavaScript.")
st.write("Use hover to see the research point name.")

