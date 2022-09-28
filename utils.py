import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DefaultConfig:

    movies = "./movies_data/movies.csv"
    ratings = "./movies_data/ratings.csv"

def _clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)

# core search logic
def search(movies, tfidf, vectorizer, title, topk):
    title = _clean_title(title)
    query_vectorizer = vectorizer.transform([title])

    #Computing cosine similarity between given title and the above calculated tf-idf matrix
    similarity = cosine_similarity(query_vectorizer,tfidf).flatten()

    indices = np.argpartition(similarity,-topk)[-topk:]
    results = movies.iloc[indices][::-1].reset_index(drop=True)
    #results.drop(["cleaned_title", "movieId"], inplace=True, axis=1)
    return results

def get_vectorizer(movies):
    vectorizer = TfidfVectorizer(ngram_range = (1,2))
    tfidf = vectorizer.fit_transform(movies['cleaned_title'])
    return vectorizer, tfidf

def add_clean_title(movies):
    movies['cleaned_title'] = movies['title'].apply(_clean_title)