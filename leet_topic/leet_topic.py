import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import math
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import logging
import warnings
from annoy import AnnoyIndex

from .bokeh_app import create_html

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def create_labels(df, document_field, encoding_model,
                  umap_params={"n_neighbors": 50, "min_dist": 0.01, "metric": 'correlation'},
                  hdbscan_params={"min_samples": 10, "min_cluster_size": 50},
                  doc_embeddings=None):
    # print(type(doc_embeddings))
    if str(type(doc_embeddings)) == "<class 'NoneType'>":
        #Load Transformer Model
        model = SentenceTransformer(encoding_model)

        #Create Document Embeddings
        logging.info("Encoding Documents")
        doc_embeddings = model.encode(df[document_field])
        logging.info("Saving Embeddings")
        np.save("embeddings", doc_embeddings)

    #Create UMAP Projection
    logging.info("Creating UMAP Projections")
    umap_proj = umap.UMAP(**umap_params).fit_transform(doc_embeddings)

    #Create HDBScan Label
    logging.info("Finding Clusters with HDBScan")
    hdbscan_labels = hdbscan.HDBSCAN(**hdbscan_params).fit_predict(umap_proj)
    df["x"] = umap_proj[:,0]
    df["y"] = umap_proj[:,1]
    df["hdbscan_labels"] = hdbscan_labels

    return df, doc_embeddings

def find_centers(df):
    #Get coordinates for each document in each topic
    topic_data = {}
    for i, topic in enumerate(df.hdbscan_labels.tolist()):
        if topic != -1:
            if topic not in topic_data:
                topic_data[topic] = {"center": [], "coords": []}
            topic_data[topic]["coords"].append((df["x"][i], df["y"][i]))

    #Calculate the center of the topic
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
    logging.info(f"{df.hdbscan_labels.tolist().count(-1)} Outliers reduced to {leet_labels.count(-1)}")
    return df


def create_tfidf(df, topic_data, document_field, spacy_model):
    nlp = spacy.load(spacy_model, disable=["ner", "attribute_ruler", "tagger", "parser"])
    lemma_docs = [" ".join([token.lemma_.lower() for token in nlp(text) if token.text not in string.punctuation]) for text in df[document_field].tolist()]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(lemma_docs)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    tfidf_df = pd.DataFrame(denselist, columns=feature_names)

    top_n = 10
    tfidf_words = []
    for vector in vectors:
        top_words = (sorted(list(zip(vectorizer.get_feature_names_out(), vector.sum(0).getA1())), key=lambda x: x[1], reverse=True)[:top_n])
        tfidf_words.append(top_words)
    df["top_words"] = tfidf_words

    if df.leet_labels.tolist().count(-1) > 0:
        topic_data[-1] = {}
    for leet_label, lemmas in zip(df.leet_labels.tolist(), lemma_docs):
        if "doc_lemmas" not in topic_data[leet_label]:
            topic_data[leet_label]["doc_lemmas"] = []
        topic_data[leet_label]["doc_lemmas"].append(lemmas)

    for leet_label, data in topic_data.items():
        # Apply the transformation using the already fitted vectorizer
        X = vectorizer.transform(data["doc_lemmas"])
        words = (sorted(list(zip(vectorizer.get_feature_names_out(), X.sum(0).getA1())), key=lambda x: x[1], reverse=True)[:top_n])
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


def download_spacy_model(spacy_model):
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        print(f'Downloading language model ({spacy_model}) for the spaCy POS tagger\n'
            "(don't worry, this will only happen once)")
        from spacy.cli import download
        download(spacy_model)

def create_annoy(doc_embeddings,
                annoy_filename="annoy_index.ann",
                annoy_branches=10,
                annoy_metric="angular"
                ):

    t = AnnoyIndex(doc_embeddings.shape[1], annoy_metric)
    for idx, embedding in enumerate(doc_embeddings):
        t.add_item(idx, embedding)

    t.build(annoy_branches)
    if ".ann" not in annoy_filename:
        annoy_filename = annoy_filename+".ann"
    t.save(annoy_filename)

    return t


def LeetTopic(df: pd.DataFrame,
            document_field: str,
            html_filename: str,
            extra_fields=[],
            max_distance=.5,
            tf_idf = False,
            spacy_model="en_core_web_sm",
            encoding_model='all-MiniLM-L6-v2',
            save_embeddings=True,
            doc_embeddings = None,
            umap_params={"n_neighbors": 50, "min_dist": 0.01, "metric": 'correlation'},
            hdbscan_params={"min_samples": 10, "min_cluster_size": 50},
            app_name="",
            build_annoy=False,
            annoy_filename="annoy_index.ann",
            annoy_branches=10,
            annoy_metric="angular"
            ):
    """
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains at least one field that are the documents you wish to model
    document_field: str
        a string that is the name of the column in which the documents in the DataFrame sit
    html_filename: str
        the name of the html file that will be created by the LeetTopic pipeline
    extra_fields: list of str (Optional)
        These are the names of the columns you wish to include in the Bokeh application.
    max_distance: float (Optional default .5)
        The maximum distance an outlier document can be to the nearest topic vector to be assigned
    spacy_model: str (Optional default en_core_web_sm)
        the spaCy language model you will use for lemmatization
    encoding_model: str (Optional default all-MiniLM-L6-v2)
        the sentence transformers model that you wish to use to encode your documents
    umap_params: dict (Optional default {"n_neighbors": 50, "min_dist": 0.01, "metric": 'correlation'})
        dictionary of keys to UMAP params and values for those params
    hdbscan_params: dict (Optional default {"min_samples": 10, "min_cluster_size": 50})
        dictionary of keys to HBDscan params and values for those params
    app_name: str (Optional)
        title of your Bokeh application
    Returns
    ----------
    df: pd.DataFrame
        This is the new dataframe that contains the metadata generated from the LeetTopic pipeline
    topic_data: dict
        This is topic-centric data generated by the LeetTopic pipeline
    """

    download_spacy_model(spacy_model)

    df, doc_embeddings = create_labels(df, document_field,
                    encoding_model, doc_embeddings=doc_embeddings,
                    umap_params=umap_params, hdbscan_params=hdbscan_params)
    logging.info("Calculating the Center of the Topic Clusters")
    topic_data = find_centers(df)
    logging.info(f"Recalculating clusters based on a max distance of {max_distance} from any topic vector")
    df = get_leet_labels(df, topic_data, max_distance)


    if tf_idf==True:
        logging.info("Creating TF-IDF representation for documents")
        df, topic_data = create_tfidf(df, topic_data, document_field, spacy_model)

    logging.info("Creating Topic Relevance")
    df, topic_data = calculate_topic_relevance(df, topic_data)

    logging.info("Generating custom Bokeh application")
    create_html(df,
                document_field=document_field,
                topic_field="leet_labels",
                html_filename=html_filename,
                topic_data=topic_data,
                tf_idf=tf_idf,
                extra_fields=extra_fields,
                app_name=app_name,
               )
    df = df.drop("selected", axis=1)

    if build_annoy == True:
        logging.info(f"Building an Annoy Index and saving it to {annoy_filename}")
        annoy_index = create_annoy(doc_embeddings,
                    annoy_filename=annoy_filename,
                    annoy_branches=annoy_branches,
                    annoy_metric=annoy_metric)
        return df, topic_data, annoy_index

    return df, topic_data