import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup
scaler = StandardScaler()
n_neighbors = 5

# Christmas Theme CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        background-image: url('https://www.transparenttextures.com/patterns/snow.png');
    }
    .main-title {
        color: #d72638;
        font-family: "Comic Sans MS", cursive, sans-serif;
        text-align: center;
    }
    .header, .expander-header {
        color: #006400;
    }
    .footer {
        color: #2e8b57;
        font-size: small;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with Christmas Cheer
st.markdown("<h1 class='main-title'>🎄 Paper Recommendation System 🎅</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='header'>Bringing you academic gifts this holiday season! 🎁</h2>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

data = load_data("outnow.csv")
affils = load_data("DATA_CSV/affils.csv")
affirelations = load_data("DATA_CSV/affi_relation.csv")

# Load TF-IDF model
with open('models/tfidf.model', 'rb') as file:
    tfidf = pickle.load(file)

# Sidebar selection
subjects = data.columns[5:32]
selected_subject = st.sidebar.selectbox("🎁 Select a Subject:", subjects)

# PCA processing
s_data = data[data[selected_subject] == 1]
subject_data = s_data.drop(columns=['title'])
titles = s_data['title']
numeric = subject_data.drop(columns=['keywords', 'affiliation_id', 'cited_by_count'])
features_scaled = scaler.fit_transform(numeric)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Add PCA results to data
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]
subject_data["Title"] = titles
subject_data["hover_text"] = subject_data["Title"].apply(
    lambda x: "<br>".join([x[i:i+50] for i in range(0, len(x), 50)])
)

# Scatter plot with Christmas colors
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    title=f"{selected_subject} Data",
    hover_name="hover_text",
    color_discrete_sequence=["#d72638", "#88d498", "#ffcc00"]
)
st.plotly_chart(fig)

# Recommendation logic
selected_points = plotly_events(fig)

def get_paper(index, df):
    return df.iloc[int(index)]

if selected_points:
    selected_index = selected_points[0]['pointIndex']
    selected_paper = get_paper(selected_index, s_data)
    with st.expander(f"🎅 Selected Paper: {selected_paper['title']}"):
        cleaned_kw = selected_paper['keywords'].replace(";", ", ")
        st.write(f"🎄 Keywords: {cleaned_kw}")
        affi_id = selected_paper["affiliation_id"]
        affiliated_info = []
        for affi in affirelations[affirelations["id"] == affi_id]["affi"]:
            aff_info = affils.iloc[affi]
            affiliated_info.append(f"{aff_info['name']}, {aff_info['country']}, {aff_info['city']}")
        st.write("🎁 Affiliated with:")
        for aff in affiliated_info:
            st.write(f"- {aff}")

    # KNN Recommendations
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='cosine')
    knn.fit(features_scaled)
    distances, indices = knn.kneighbors(features_scaled[selected_index].reshape(1, -1))
    rec_papers, cited_counts = [], []

    for i, idx in enumerate(indices[0][1:]):  # Exclude the selected paper
        rec_paper = get_paper(idx, s_data)
        rec_papers.append(f"Recommend #{i+1}: {rec_paper['title']}")
        cited_counts.append(rec_paper["cited_by_count"])
        with st.expander(f"🎄 Recommend #{i+1}: {rec_paper['title']}"):
            st.write(f"🎁 Keywords: {rec_paper['keywords'].replace(';', ', ')}")
            st.write(f"Distance: {distances[0][i+1]:.4f}")
            st.write(f"Cited by count: {rec_paper['cited_by_count']}")
    
    # Bar chart of recommended papers
    rec_df = pd.DataFrame({
        "Recommended Paper": rec_papers,
        "Cited by Count": cited_counts
    })
    fig_rec = px.bar(
        rec_df, x="Recommended Paper", y="Cited by Count",
        title="🎄 Cited by Count of Recommended Papers 🎁",
        color="Cited by Count", color_continuous_scale="reds"
    )
    st.plotly_chart(fig_rec)

else:
    st.markdown("<h4 style='color:#d72638'>Please select a paper to see recommendations! 🎁</h4>", unsafe_allow_html=True)

# Footer with holiday greetings
st.markdown("<div class='footer'>Happy Holidays and Happy Learning! 🎄🎓</div>", unsafe_allow_html=True)
