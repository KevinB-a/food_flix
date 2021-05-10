from numpy import empty
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
from IPython.display import Image, HTML
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
sys.path.insert(0,"/home/apprenant/simplon_project/food_flix")
from src.streamlit_functions import *
from fuzzywuzzy import fuzz
from .streamlit_const import *

#--------------------------------------------#
#            Import data                     #
#--------------------------------------------#
df = pd.read_csv('/home/apprenant/simplon_project/food_flix/data/df_clean.csv', usecols = ['id','product_name','ingredients_text','countries', 'nutrition_grade_fr'])

#st.dataframe(df)
#--------------------------------------------#
#                Header                      #
#--------------------------------------------#
st.title('moteur de recommandation basée sur le contenu')
#--------------------------------------------#
#                Sidebar                     #
#--------------------------------------------#
st.sidebar.title('moteur de recommandation')
method = st.sidebar.radio('methode de recherche', ['TF-IDF', 'CountVectorizer', 'BERT'])
user_value = st.sidebar.text_input('recherche')
st.sidebar.slider('donnez le nombre de valeurs lié à la recherche', min_value = 1 , max_value = 10)

st.markdown('les resultats s\'affiche ici')
#--------------------------------------------#
#                  user_value                #
#--------------------------------------------#
if user_value : 
    results_are_shown = True
    empty = st.empty()
    if df["product_name"].to_string().find(user_value) == -1 or df["brands"].to_string().find(user_value) == -1:
        with empty.beta_container():
            results_are_shown = False
            fuzzies = find_fuzzy(user_value, df["product_name"].to_list())
            st.warning(WARN_INPUT_NOT_FOUND.format(user_value))
            choices = [fuzzy[0] for fuzzy in fuzzies]
            choices.insert(0, "")
            choices = list(set(choices))
            choices.sort()
            radio = st.radio("", choices)
            if radio != "":
                user_value = radio
                results_are_shown = True
                empty.empty()
                print(user_value, type(user_value))
#--------------------------------------------#
#                  Methods                   #
#--------------------------------------------#
if method == 'TF-IDF' : 
    model = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0)
elif method == 'CountVectorizer' : 
    model = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0)

 
