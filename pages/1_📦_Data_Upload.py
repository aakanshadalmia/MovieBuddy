from urllib import request
import streamlit as st
import pandas as pd
from utils import DefaultConfig

st.set_page_config(layout="wide")

default_config = DefaultConfig()

st.title("Data Upload")

movies = None
ratings = None
with st.spinner("The engine is loading. Please wait..."):
    movies = pd.read_csv(default_config.movies)
    ratings = pd.read_csv(default_config.ratings)

st.write(
    """
    By default the engine uses Movielens 25m dataset (samples for which are shown below).
    If you want to try out the engine with your custom dataset upload the files with the proper schema and enjoy!!
    """
)

movies_col, ratings_col = st.columns(2)
with movies_col:
    st.dataframe(movies.head(20))

with ratings_col:
    st.dataframe(ratings.head(20))


use_default = st.radio("Upload custom data?", ["Yes", "No"], index=1, horizontal=True)

if use_default == 'Yes':
    movies_file = st.file_uploader("Movies.csv")
    ratings_file = st.file_uploader("Ratings.csv")
    if movies_file:
        movies = pd.read_csv(movies_file)
    if ratings_file:
        ratings = pd.read_csv(ratings_file)

st.session_state["movies"] = movies
st.session_state["ratings"] = ratings

st.write("Head over to the search and recommendation sections now!")