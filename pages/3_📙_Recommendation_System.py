import pandas as pd
import streamlit as st
from utils import DefaultConfig, get_vectorizer, search, add_clean_title

st.set_page_config(layout="wide")

default_config = DefaultConfig()

st.title("Let's get some Recommendations!")

if st.session_state.get("movies", None) is None or st.session_state.get("ratings", None) is None:
    with st.spinner("The system is loading. Please wait..."):
        movies = pd.read_csv(default_config.movies)
        ratings = pd.read_csv(default_config.ratings)
        st.session_state["movies"] = movies
        st.session_state["ratings"] = ratings
movies = st.session_state["movies"]
ratings = st.session_state["ratings"]

add_clean_title(movies)


# core recommendation logic
def _collaborative_recs(movie_id):
    similar_people = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4.0)]['userId'].unique()

    similar_people_movie_recs = ratings[(ratings["userId"].isin(similar_people)) & (ratings['rating'] > 4.0)]

    similar_movie_recs = similar_people_movie_recs['movieId']
    similar_movie_recs = similar_movie_recs.value_counts() / len(similar_people)
    similar_movie_recs = similar_movie_recs[similar_movie_recs > 0.1]

    all_people = ratings[(ratings['movieId'].isin(similar_movie_recs.index)) & (ratings['rating'] > 4.0)]
    all_people_recs = all_people['movieId'].value_counts() / len(all_people['userId'].unique())

    rec_percentage = pd.concat([similar_movie_recs,all_people_recs],axis = 1)
    rec_percentage.columns = ["Similar","All"]
    rec_percentage["Similarity Score"] = rec_percentage["Similar"] / rec_percentage["All"]
    rec_percentage = rec_percentage.sort_values("Similarity Score",ascending = False)

    final_recommendations = rec_percentage.merge(movies, left_index = True, right_on = 'movieId')
    final_recommendations = final_recommendations[['Similarity Score','title','genres']][:10:]

    return final_recommendations.reset_index(drop=True)

query_movie = st.text_input("Enter a movie you like:").strip()
vectorizer, tfidf = get_vectorizer(movies)

if query_movie != "":
    topk = 1
    results = search(movies, tfidf, vectorizer, query_movie, topk)
    movie_id = results.iloc[0]["movieId"]

    recommendations = _collaborative_recs(movie_id)
    st.write("""**Here are some movies you might like!**""")
    st.dataframe(recommendations[['title','genres']], use_container_width=True)
