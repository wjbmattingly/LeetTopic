# Libraries
import numpy as np
import pandas as pd
import pickle
from leet_topic import leet_topic
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import random

stopwords_list = []
stopwords_list += stopwords.words('french')
stopwords_list += stopwords.words('english')
stopwords_list += ["dysfonctionnement", "plus", "accès", "pas", "impossible"]

# Functions
def remove_stopwords(text):
    # Split the text into words
    words = text.split()

    # Remove the stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords_list]

    # Rejoin the words into a single string
    filtered_text = " ".join(filtered_words)

    return filtered_text

# Data
with open("embeddings", "rb") as fichier:
    precomputed_embeddings = pickle.load(fichier)

data = pd.read_csv("data.csv", encoding="utf-8")["Description courte"].to_list()
data = [d.lower() for d in data if type(d)==str]
data = [remove_stopwords(d) for d in tqdm(data)]
df = pd.DataFrame(data, columns=["descriptions"])

# LeetTopic
leet_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="descriptions",
                                          html_filename="demo.html",
                                          spacy_model="fr_core_news_md",
                                          encoding_model="C:/Users/equinetpa/Desktop/Poursuite/pipeline/models/camembert-base",
                                          embeddings = precomputed_embeddings
                                          )
