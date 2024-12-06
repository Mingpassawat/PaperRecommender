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
scaler = StandardScaler()

n_neighbors = 5

#Christmas Theme CSS
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
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

    </style>
    """, unsafe_allow_html=True)



subject_codes_swapped = {
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




# Load dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)


ok = st.sidebar.checkbox("Decrease vector dimension")


if ok:
    data = load_data("outnow.csv").drop(columns=["Unnamed: 0"])
else:
    data = load_data("outnow_3.csv").drop(columns=["Unnamed: 0"])
data2 = load_data("outnow_2.csv").drop(columns=["Unnamed: 0"])
print(data.info())
print(data2.info())
affils = load_data("DATA_CSV/affils.csv")
affirelations = load_data("DATA_CSV/affi_relation.csv")

# Load tfidf model
with open('models/tfidf.model', 'rb') as file:
    tfidf = pickle.load(file)

# Columns and scaling
subjects = subject_codes_swapped.keys()
scaler = StandardScaler()

# Sidebar selection
subject_list = subjects
selected_subject = subject_codes_swapped[st.sidebar.selectbox("🎁 Select a subject:", subject_list)]

# # Title
# st.title("Paper Recommendation System")

# Title with Christmas Cheer
st.markdown("<h1 class='main-title'>🎄 Paper Recommendation System 🎅</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='header'>Bringing you academic gifts this holiday season! 🎁</h2>", unsafe_allow_html=True)

st.header("Using KNN + Word2Vec")


# Filter and transform data for PCA
s_data = data[data[selected_subject] == 1]
if ok:
    subject_data = data[data[selected_subject] == 1].drop(columns = ['title']) # Important
else:
    subject_data = data[data[selected_subject] == 1].drop(columns = ['title', 'title_keywords', 'token']) # Important
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
    lambda x: "<br>".join([x[i:i+50] for i in range(0, len(x), 50)])  # Break lines every 30 characters
)

# Plotly scatter plot
fig = px.scatter(
    subject_data,
    x="PCA1",
    y="PCA2",
    title=f"{selected_subject} Data",
    hover_name="hover_text",  # Use the hover_text column directly
    color_discrete_sequence=["#d72638", "#88d498", "#ffcc00"]
)
fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><extra></extra>", customdata=subject_data[["hover_text"]])

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

if ok:
    st.write("Vector dimension: 5")
else:
    st.write("Vector dimension: 100")

recpaperlist1 =[]
citedcount1 = []

# Selected point expander
if selected_points:  # Check if any point is selected
    # Get the selected paper
    selected_index = selected_points[0]['pointIndex']
    selected_paper = get_paper(selected_index, s_data)

    # Display selected paper information
    with st.expander(f"🎅 Selected paper title: {selected_paper['title']}"):
        try:
            cleaned_kw = selected_paper['keywords'].replace(";", ", ")
            st.write(f"🎄 Keywords: {cleaned_kw}")
        except:
            st.write("Keyword: No keywords")
        selectedaffilist = []
        try:
            affi_id = selected_paper["affiliation_id"]
            uninumberdf = affirelations[affirelations["id"] == affi_id]
            for value in uninumberdf["affi"]:
                selectedaffilist.append(value)
            st.write("🦌 Affilated with: ")
            for i in selectedaffilist:
                try:
                    st.write(affils.iloc[i]["name"]+", "+affils.iloc[i]["country"]+", "+affils.iloc[i]["city"]+"\n")
                except:
                    st.write("Affiliation: No data")
        except:
            st.write("Affilated with: Not found")


    # Calculate recommendations using nearest neighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='cosine')  # Add 1 to account for the selected paper
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
        citedcount1.append(rec_paper['cited_by_count'])
        recpaperlist1.append("Recommend# "+ str(i+1))
        with st.expander(f"🎄 Recommend #{i+1}: {rec_paper['title']}"):
            try:
                cleaned_kw = rec_paper['keywords'].replace(";", ", ")
            except:
                cleaned_kw = "Not Found"
            st.write(f"🎁 Keywords: {cleaned_kw}")
            st.write(f"⛄ Distance from selected paper: {filtered_distances[i]}")
            st.write(f"🎊 Cited by count: {int(rec_paper['cited_by_count'])}")
            selectedaffilist = []
            affi_id = rec_paper["affiliation_id"]
            uninumberdf = affirelations[affirelations["id"] == affi_id]
            for value in uninumberdf["affi"]:
                selectedaffilist.append(value)
            st.write("🦌 Affilated with: ")
            try:
                st.write(affils.iloc[i]["name"]+", "+affils.iloc[i]["country"]+", "+affils.iloc[i]["city"]+"\n")
            except:
                st.write("Affiliate: No data")
    data1 ={
    "Recommended Paper": recpaperlist1,
    "Cited by Count" : citedcount1
    }
    df = pd.DataFrame(data1)
    fig = px.bar(df, x="Recommended Paper", y="Cited by Count", title="Cited by Count of Recommended Papers")
    st.plotly_chart(fig)   
else:
    st.write("Please select a paper")




st.header("Using KNN + TF-IDF")

recpaperlist2 =[]
citedcount2 = []

if selected_points:  # Check if any point is selected
    # Retrieve the selected paper's index and information
    selected_index = selected_points[0]['pointIndex']
    selected_paper = get_paper(selected_index, s_data)

    with st.expander(f"🎅 Selected paper title: {selected_paper['title']}"):
        cleaned_kw = selected_paper['keywords'].replace(";", ", ")
        st.write(f"🎄 Keywords: {cleaned_kw}")
        selectedaffilist = []
        affi_id = selected_paper["affiliation_id"]
        uninumberdf = affirelations[affirelations["id"] == affi_id]
        for value in uninumberdf["affi"]:
            selectedaffilist.append(value)
        st.write("🦌 Affilated with: ")
        for i in selectedaffilist:
            try:
                st.write(affils.iloc[i]["name"]+", "+affils.iloc[i]["country"]+", "+affils.iloc[i]["city"]+"\n")
            except:
                st.write("Affiliate: No data")


    # TF-IDF Vectorization
    s_data2 = data2[data2[selected_subject] == 1]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(s_data2["title_keywords"])

    # Calculate TF-IDF nearest neighbors
    tfidf_knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='cosine')
    tfidf_knn.fit(X_tfidf)

    # Find neighbors for the selected paper
    selected_tfidf_features = X_tfidf[selected_index]
    distances, indices = tfidf_knn.kneighbors(selected_tfidf_features)

    # Filter out the selected paper from the recommendations
    filtered_indices_tfidf = [idx for idx in indices[0] if idx != selected_index]
    filtered_distances_tfidf = [distances[0][i] for i, idx in enumerate(indices[0]) if idx != selected_index]

    # Display recommended papers based on TF-IDF
    for i, index in enumerate(filtered_indices_tfidf[:n_neighbors]):  # Limit to `n_neighbors`
        rec_paper = get_paper(index, s_data2)
        citedcount2.append(rec_paper['cited_by_count'])
        recpaperlist2.append("Recommend# "+ str(i+1))
        with st.expander(f"🎄 Recommend #{i+1}: {rec_paper['title']}"):
            if rec_paper['keywords'] and isinstance(rec_paper['keywords'], str):
                cleaned_kw = rec_paper['keywords'].replace(";", ", ")
                st.write(f"🎁 Keywords: {cleaned_kw}")
            st.write(f"⛄ Distance from selected paper: {filtered_distances_tfidf[i]:.4f}")
            st.write(f"🎊 Cited by count: {int(rec_paper['cited_by_count'])}")
            selectedaffilist = []
            affi_id = rec_paper["affiliation_id"]
            uninumberdf = affirelations[affirelations["id"] == affi_id]
            for value in uninumberdf["affi"]:
                selectedaffilist.append(value)
            st.write("🦌 Affilated with: ")
            for i in selectedaffilist:
                try:   
                    st.write(affils.iloc[i]["name"]+", "+affils.iloc[i]["country"]+", "+affils.iloc[i]["city"]+"\n")
                except:
                    st.write("Affiliate: No data")

    data2 ={
    "Recommended Paper": recpaperlist2,
    "Cited by Count" : citedcount2
    }
    df = pd.DataFrame(data2)
    fig = px.bar(df, x="Recommended Paper", y="Cited by Count", title="Cited by Count of Recommended Papers")
    st.plotly_chart(fig)
else:
    st.write("Please select a paper")
    
st.markdown("<div class='footer'>Happy Holidays and Happy Learning! 🎄🎓</div>", unsafe_allow_html=True)