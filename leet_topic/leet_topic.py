import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import math
from annoy import AnnoyIndex
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim.corpora as corpora

def create_labels(df):

    #Load Transformer Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Create Document Embeddings
    doc_embeddings = model.encode(df.documents)

    #Create UMAP Projection
    umap_proj = umap.UMAP(n_neighbors=50,
                              min_dist=0.01,
                              metric='correlation').fit_transform(doc_embeddings)

    #Create HDBScan Label
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2).fit_predict(umap_proj)
    df["x"] = umap_proj[:,0]
    df["y"] = umap_proj[:,1]
    df["hdbscan_labels"] = hdbscan_labels
    # outlier = [True if topic == -1 else False for topic in hdbscan_labels]
    # df["outlier"] = outlier
    return df

def find_centers(df):
    #Get coordinates for each
    topic_data = {}
    for i, topic in enumerate(df.hdbscan_labels.tolist()):
        if topic != -1:
            if topic not in topic_data:
                topic_data[topic] = {"center": [], "coords": []}
            topic_data[topic]["coords"].append((df["x"][i], df["y"][i]))

    #Calculate Center
    for topic, data in topic_data.items():
        x = [coord[0] for coord in data["coords"]]
        y = [coord[1] for coord in data["coords"]]
        c = (x, y)
        topic_data[topic]["center"] = (sum(c[0])/len(c[0]),sum(c[1])/len(c[1]))
    return topic_data

def get_leet_labels(df, topic_data, max_distance):
    # Get New Topic Numbers
    leet_labels = []
    for i, topic in enumerate(df.hdbscan_labels.tolist()):
        if topic == -1:
            closest = -1
            distance = max_distance
            for topic_num, coords in topic_data.items():
                center = coords["center"]
                current_distance = math.dist(center, (df["x"][i], df["y"][i]))
                if current_distance < max_distance and current_distance < distance:
                    closest = topic_num
                    distance = current_distance
            leet_labels.append(closest)
        else:
            leet_labels.append(topic)
    df["leet_labels"] = leet_labels
    print(f"{df.hdbscan_labels.tolist().count(-1)} Outliers reduced to {leet_labels.count(-1)}")
    return df


def create_tfidf(df, topic_data):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    lemma_docs = []
    for text in df.documents.tolist():
        text = "".join([c for c in text if c.isdigit() == False])
        doc = nlp(text)
        lemma_docs.append(" ".join([token.lemma_.lower() for token in doc if token.pos_ != "PUNCT"]))

    df["lemma_docs"] = lemma_docs
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(lemma_docs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    tfidf_df = pd.DataFrame(denselist, columns=feature_names)


    top_n = 10
    tfidf_words = []
    for vector in vectors:
        top_words = (sorted(list(zip(vectorizer.get_feature_names(),
                                                     vector.sum(0).getA1())),
                                         key=lambda x: x[1], reverse=True)[:top_n])
        tfidf_words.append(top_words)
    df["top_words"] = tfidf_words
    if df.leet_labels.tolist().count(-1) > 0:
        topic_data[-1] = {}
    for leet_label, lemmas in zip(df.leet_labels.tolist(), lemma_docs):
        if "doc_lemmas" not in topic_data[leet_label]:
            topic_data[leet_label]["doc_lemmas"] = []
        topic_data[leet_label]["doc_lemmas"].append(lemmas)


    for leet_label, data in topic_data.items():
        X = vectorizer.fit_transform(data["doc_lemmas"])
        words = (sorted(list(zip(vectorizer.get_feature_names(),
                                                     X.sum(0).getA1())),
                                         key=lambda x: x[1], reverse=True)[:top_n])
        topic_data[leet_label]["key_words"] = words
    return df, topic_data

def calculate_topic_relevance(df, topic_data):
    rel2topic = []
    for idx, row in df.iterrows():
        topic_num = row.leet_labels
        if topic_num != -1:
            if "relevance_docs" not in topic_data[topic_num]:
                topic_data[topic_num]["relevance_docs"] = []
            score = math.dist(topic_data[topic_num]["center"], (row["x"], row["y"]))
            rel2topic.append(score)
            topic_data[topic_num]["relevance_docs"].append((idx, score))
        else:
            rel2topic.append((idx, 0))
    for topic_num, data in topic_data.items():
        if topic_num != -1:
            data["relevance_docs"].sort(key = lambda x: x[1])
            data["relevance_docs"].reverse()
    return df, topic_data


def LeetTopic(documents, max_distance=.5):
    df = pd.DataFrame()
    df["documents"] = documents

    df = create_labels(df)
    topic_data = find_centers(df)
    df = get_leet_labels(df, topic_data, max_distance)
    df, topic_data = create_tfidf(df, topic_data)
    df, topic_data = calculate_topic_relevance(df, topic_data)
    return df, topic_data
























        #
