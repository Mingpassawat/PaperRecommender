import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Streamlit app setup
st.title("k-NN Visualization for Subjects")

# Input: User selects the subject



# Example DataFrame (replace with your actual DataFrame)
# The DataFrame must have a column for each subject (e.g., 'MED', 'ARTS', etc.) and features for k-NN
df_expanded = pd.read_csv("out.csv")  # Example data
subjects = df_expanded.columns
selected_subject = st.selectbox("Select a subject:", subjects)

# Filter and scale features based on the selected subject
scaler = StandardScaler()
if selected_subject in df_expanded.columns:
    features = df_expanded.loc[df_expanded[selected_subject] == 1]
    features_scaled = scaler.fit_transform(features)

    # k-NN setup
    n_neighbors = 5
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute')
    knn.fit(features_scaled)
    distances, indices = knn.kneighbors([features_scaled[0]])

    # PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], label="Data Points", alpha=0.6)
    query_point = features_2d[0]
    neighbors = features_2d[indices[0]]
    plt.scatter(query_point[0], query_point[1], color='red', label='Query Point', s=100)
    plt.scatter(neighbors[:, 0], neighbors[:, 1], color='green', label='Neighbors', s=100)
    plt.title(f'k-NN Visualization for {selected_subject}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    st.pyplot(plt.gcf())
else:
    st.error("Selected subject is not found in the dataset.")