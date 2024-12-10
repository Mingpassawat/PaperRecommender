import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_plotly_events import plotly_events
from scipy.spatial import KDTree
import pickle

# Initialize constants
N_NEIGHBORS = 5
SCALER = StandardScaler()
SUBJECT_CODES_SWAPPED = {
    "Agricultural and Biological Sciences": "AGRI",
    "Arts and Humanities": "ARTS",
    "Biochemistry, Genetics and Molecular Biology": "BIOC",
    "Business, Management and Accounting": "BUSI",
    "Chemical Engineering": "CENG",
    "Chemistry": "CHEM",
    "Computer Science": "COMP",
    "Decision Sciences": "DECI",
    "Dentistry": "DENT",
    "Earth and Planetary Sciences": "EART",
    "Economics, Econometrics and Finance": "ECON",
    "Energy": "ENER",
    "Engineering": "ENGI",
    "Environmental Science": "ENVI",
    "Health Professions": "HEAL",
    "Immunology and Microbiology": "IMMU",
    "Materials Science": "MATE",
    "Mathematics": "MATH",
    "Medicine": "MEDI",
    "Neuroscience": "NEUR",
    "Nursing": "NURS",
    "Pharmacology, Toxicology and Pharmaceutics": "PHAR",
    "Physics and Astronomy": "PHYS",
    "Psychology": "PSYC",
    "Social Sciences": "SOCI",
    "Veterinary": "VETE",
    "Multidisciplinary": "MULT"
}

# Load datasets
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Christmas Theme CSS
st.markdown(
    """
    <style>
        body {
            background-color: #D2B48C;
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
        .stApp {
            background-color: #D2B48C;
            background-size: cover;
        }
        [data-testid="stSidebar"] {
            background-color: #FFF5E1;
            padding: 20px;
            font-family: 'Arial';
        }
        [data-testid="stSidebar"] h1 {
            color: #1E90FF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar options
decrease_vector_dimension = st.sidebar.checkbox("Decrease vector dimension")
search_query = st.sidebar.text_input("🔍 Search for papers by title:")

# Load data based on user input
if decrease_vector_dimension:
    data = load_data("outnow.csv").drop(columns=["Unnamed: 0"])
else:
    data = load_data("outnow_3.csv").drop(columns=["Unnamed: 0"])

data2 = load_data("outnow_2.csv").drop(columns=["Unnamed: 0"])
affils = load_data("DATA_CSV/affils.csv")
affirelations = load_data("DATA_CSV/affi_relation.csv")

# Load TF-IDF model
with open('models/tfidf.model', 'rb') as file:
    tfidf_model = pickle.load(file)

# Sidebar subject selection
subjects = list(SUBJECT_CODES_SWAPPED.keys())
selected_subject = SUBJECT_CODES_SWAPPED[st.sidebar.selectbox("🎁 Select a subject:", subjects)]

# Main title
st.markdown("<h1 class='main-title'>🎄 Paper Recommendation System 🎅</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='header'>Bringing you academic gifts this holiday season! 🎁</h2>", unsafe_allow_html=True)

# Filter data based on subject
subject_data = data[data[selected_subject] == 1]

# Apply search filter
if search_query:
    subject_data = subject_data[subject_data['title'].str.contains(search_query, case=False, na=False)]

# Check if data exists after filtering
if subject_data.empty:
    st.warning("No data available for the selected subject or filter. Please adjust your filters.")
    st.stop()

# Prepare data for PCA
titles = subject_data['title']
numeric = subject_data.drop(columns=['title', 'keywords', 'affiliation_id', 'cited_by_count'], errors='ignore')
features_scaled = SCALER.fit_transform(numeric)

# PCA
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)
subject_data["PCA1"] = features_2d[:, 0]
subject_data["PCA2"] = features_2d[:, 1]
subject_data["Title"] = titles

# Scatter plot
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    title=f"{selected_subject} Data",
    hover_name="Title",
    color_discrete_sequence=["#d72638", "#88d498", "#ffcc00"]
)
selected_points = plotly_events(fig)
st.plotly_chart(fig)

# KNN Recommendations
if selected_points:
    selected_index = selected_points[0]['pointIndex']
    selected_paper = subject_data.iloc[selected_index]

    # Nearest neighbors with PCA features
    knn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='cosine')
    knn.fit(features_scaled)
    distances, indices = knn.kneighbors(features_scaled[selected_index].reshape(1, -1))

    # Display recommendations
    for i, idx in enumerate(indices[0][1:N_NEIGHBORS + 1]):
        rec_paper = subject_data.iloc[idx]
        with st.expander(f"🎄 Recommendation #{i + 1}: {rec_paper['Title']}"):
            st.write(f"🎁 Keywords: {rec_paper.get('keywords', 'No keywords')}")
            st.write(f"⛄ Distance: {distances[0][i + 1]:.4f}")
            st.write(f"🎊 Cited by count: {rec_paper.get('cited_by_count', 'Unknown')}")
else:
    st.write("Please select a paper.")

st.markdown("<div class='footer'>Happy Holidays and Happy Learning! 🎄🎓</div>", unsafe_allow_html=True)
