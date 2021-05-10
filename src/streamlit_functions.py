# this file contains functions to use for the application
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from fuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from app.streamlit_const import STOPS
from sklearn.metrics.pairwise import linear_kernel


def find_fuzzy(x, series):
    return process.extractBests(x, series, limit=10)

def find_closest(model, X, query):
    x = model.transform([query])
    cosine_similarities = linear_kernel(x, X)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    return [i for i in similar_indices]

