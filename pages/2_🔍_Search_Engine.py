import re
import pandas as pd
import streamlit as st
from utils import DefaultConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import search, add_clean_title, get_vectorizer


st.set_page_config(layout="wide")

default_config = DefaultConfig()

st.title("Search for a Movie")

if st.session_state.get("movies", None) is None or st.session_state.get("ratings", None) is None:
    with st.spinner("The engine is loading. Please wait..."):
        movies = pd.read_csv(default_config.movies)
        ratings = pd.read_csv(default_config.ratings)
        st.session_state["movies"] = movies
        st.session_state["ratings"] = ratings
movies = st.session_state["movies"]
ratings = st.session_state["ratings"]


movies = st.session_state["movies"]
add_clean_title(movies)

vectorizer, tfidf = get_vectorizer(movies)

query_movie = st.text_input("Enter movie name / keywords:", value="").strip()
if query_movie != "":
    topk = st.slider("Number of relevant outputs?", 1, 10, value=5)
    results = search(movies, tfidf, vectorizer, query_movie, topk)
    st.write("""**Here are some movie names we found :smile:**""")
    st.dataframe(results[['title','genres']], use_container_width=True)
