import streamlit as st

st.set_page_config(layout="wide")
st.write(
    """
    # Movie Search Engine and Recommendation System

This project deals with building a movie recommendation system from scratch using user based collaborative filtering.

**Dataset used:** Movielens 25m

**Description of dataset:**
  - It describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service.
  - It contains 25000095 ratings and 1093360 tag applications across 62423 movies.
  - These data were created by 162541 users between January 09, 1995 and November 21, 2019.
  - This dataset was generated on November 21, 2019.

**Steps:**

- **Data Preprocessing:** Data has been cleaned to remove any special characters that may hamper the quality of the recommendations

- **Search Engine:** Efficient search in the huge dataset has been ensured by building a search engine that takes as input a movie name and returns the top 5 movie names with a similar title

- **Recommendation Engine:** Utilizing the concept of user based collaborative filtering, top recommendations are found using those movies that people similar to the user have liked.

- **Interactivity:** To take it one step further and ensure that results are being returned in real time for users, both the search engine and the recommendation system have been wrapped into an interactive widget. Thus, the output is dynamic allowing for an enhanced user experience."""
)
