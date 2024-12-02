import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('out.csv')

# Load and preprocess data
data = load_data()

# Extract subject columns and names
subjects = data.columns[1:28]  # Adjust column slicing as needed
research_names = data['ResearchName']  # Assuming there's a 'ResearchName' column

# Sidebar dropdown
selected_subject = st.sidebar.selectbox('Select a subject:', subjects)

# Filter data for the selected subject
filtered_data = data[(data[selected_subject] == 1) & (data["COMP"] == 1)]

if not filtered_data.empty:
    # Standardize feature columns
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(filtered_data.iloc[:, 1:])  # Adjust columns for numerical features

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    # Add PCA results back to the DataFrame
    filtered_data['PCA1'] = features_2d[:, 0]
    filtered_data['PCA2'] = features_2d[:, 1]

    # Create KDTree for nearest neighbor calculation
    tree = KDTree(features_2d)

    # Define a hover callback
    def on_hover(selected_point):
        _, indices = tree.query(selected_point, k=5)
        neighbors = filtered_data.iloc[indices]
        return neighbors['ResearchName'].tolist()

    # Plot the data with hover functionality
    st.title(f"Interactive Chart for {selected_subject}")
    fig = px.scatter(
        filtered_data,
        x='PCA1',
        y='PCA2',
        hover_name='ResearchName',
        title=f"{selected_subject} Data"
    )

    # Add hover functionality
    st.plotly_chart(fig, use_container_width=True)

    # Simulate hover functionality
    st.write("Hover over a point to see the 5 nearest neighbors.")
    selected_point_index = st.number_input("Select a point index for testing hover (0 to N):", 
                                           min_value=0, 
                                           max_value=len(features_2d)-1, 
                                           step=1, 
                                           value=0)
    selected_point = features_2d[selected_point_index]
    neighbors = on_hover(selected_point)
    st.write(f"5 Nearest Neighbors for point {selected_point_index}:")
    st.write(neighbors)
else:
    st.warning("No data available for the selected subject and conditions.")
