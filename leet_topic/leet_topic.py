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
from random import random

from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, DataTable, TableColumn, MultiChoice, HTMLTemplateFormatter, TextAreaInput
from bokeh.plotting import figure, output_file, show
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from bokeh.palettes import Category10, Cividis256, Turbo256
from bokeh.transform import linear_cmap
from typing import Tuple, Optional
import bokeh
import bokeh.transform

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#From Bulk Library
def get_color_mapping(
    df: pd.DataFrame,
    topic_field,
) -> Tuple[Optional[bokeh.transform.transform], pd.DataFrame]:
    """Creates a color mapping"""

    color_datatype = str(df[topic_field].dtype)
    if color_datatype == "object":
        df[topic_field] = df[topic_field].apply(
            lambda x: str(x) if not (type(x) == float and np.isnan(x)) else x
        )
        all_values = list(df[topic_field].dropna().unique())
        if len(all_values) == 2:
            all_values.extend([""])
        elif len(all_values) > len(Category10) + 2:
            raise ValueError(
                f"Too many classes defined, the limit for visualisation is {len(Category10) + 2}. "
                f"Got {len(all_values)}."
            )
        mapper = factor_cmap(
            field_name=topic_field,
            palette=Category10[len(all_values)],
            factors=all_values,
            nan_color="grey",
        )
    elif color_datatype.startswith("float") or color_datatype.startswith("int"):
        all_values = df[topic_field].dropna().values
        mapper = linear_cmap(
            field_name=topic_field,
            palette=Turbo256,
            low=all_values.min(),
            high=all_values.max(),
            nan_color="grey",
        )
    else:
        raise TypeError(
            f"We currently only support the following type for 'color' column: 'int*', 'float*', 'object'. "
            f"Got {color_datatype}."
        )
    return mapper, df





def create_html(df, document_field, topic_field, html_filename, extra_fields=[]):
    fields = ["x", "y", document_field, topic_field, "selected"]
    fields = fields+extra_fields
    output_file(html_filename)


    # df = pd.read_csv("demo.csv")

    mapper, df = get_color_mapping(df, topic_field)
    df['selected'] = False
    categories = df[topic_field].unique()
    categories = [str(x) for x in categories]



    s1 = ColumnDataSource(df)


    columns = [
            TableColumn(field=topic_field, title=topic_field, width=10),
            TableColumn(field=document_field, title=document_field, width=500),
    ]
    for field in extra_fields:
        columns.append(TableColumn(field=field, title=field, width=100))


    p1 = figure(width=500, height=500, tools="pan,tap,wheel_zoom,lasso_select,box_zoom,box_select,reset", active_scroll="wheel_zoom", title="Select Here", x_range=(df.x.min(), df.x.max()), y_range=(df.y.min(), df.y.max()))
    # p1.circle('x', 'y', source=s1, alpha=0.6)
    circle_kwargs = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s1,
                        # "alpha": "alpha",
                         "color": mapper
                        }
    scatter = p1.circle(**circle_kwargs)

    s2 = ColumnDataSource(data=dict(x=[], y=[]))
    p2 = figure(width=500, height=500, tools="pan,tap,lasso_select,wheel_zoom,box_zoom,box_select,reset", active_scroll="wheel_zoom", title="Analyze Selection", x_range=(df.x.min(), df.x.max()), y_range=(df.y.min(), df.y.max()))
    # p1.circle('x', 'y', source=s1, alpha=0.6)
    circle_kwargs2 = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s2,
                        # "alpha": "alpha",
                         "color": mapper
                        }
    scatter2 = p2.circle(**circle_kwargs2)

    multi_choice = MultiChoice(value=[], options=categories, title='Selection:')
    data_table = DataTable(source=s2,
                           columns=columns,
                           width=700,
                           height=500,
                          sortable=True,
                          autosize_mode='none')
    selected_texts = TextAreaInput(value = "", title = "Selected texts", width = 700, height=500)

    def field_string(field):
        return """d2['"""+field+"""'] = []\n"""

    def push_string(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][inds[i]])\n"""

    def indices_string(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][s1.selected.indices[i]])\n"""

    def push_string2(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][i])\n"""

    def list_creator(fields, str_type=""):
        main_str = ""
        for field in fields:
            if str_type == "field":
                main_str=main_str+field_string(field)
            elif str_type == "push":
                main_str=main_str+push_string(field)
            elif str_type == "indices":
                main_str=main_str+indices_string(field)
            elif str_type == "push2":
                main_str=main_str+push_string2(field)
        return main_str

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s4=multi_choice), code="""
            const inds = cb_obj.indices;
            const d1 = s1.data;
            const d2 = s2.data;
            const d4 = s4;"""+list_creator(fields=fields, str_type="field")+
            """for (let i = 0; i < inds.length; i++) {"""+
            list_creator(fields=fields, str_type="push")+

            """}
            const res = [...new Set(d2['"""+topic_field+"""'])];

            d4.value = res.map(function(e){return e.toString()});
            s1.change.emit();
            s2.change.emit();

        """)
    )


    multi_choice.js_on_change('value', CustomJS(args=dict(s1=s1, s2=s2, scatter=scatter), code="""
            let values = cb_obj.value;
            let unchange_values = cb_obj.value;
            const d1 = s1.data;
            const d2 = s2.data;
            const plot = scatter;
            s2.selected.indices = [];

            for (let i = 0; i < s1.selected.indices.length; i++) {
                for (let j =0; j < values.length; j++) {
                    if (d1."""+topic_field+"""[s1.selected.indices[i]] == values[j]) {
                        values = values.filter(item => item !== values[j]);
                    }
                }
            }

            """+list_creator(fields=fields, str_type="field")+
            """
            for (let i = 0; i < s1.selected.indices.length; i++) {
                if (unchange_values.includes(String(d1."""+topic_field+"""[s1.selected.indices[i]]))) {
                    """+
                    list_creator(fields=fields, str_type="indices")+

                    """
                }
            }

            for (let i = 0; i < d1."""+topic_field+""".length; i++) {
                if (values.includes(String(d1."""+topic_field+"""[i]))) {
                        """+
                        list_creator(fields=fields, str_type="push2")+

                        """
                }
            }

            s2.change.emit();

        """)
    )


    s2.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s_texts=selected_texts), code="""
            const inds = cb_obj.indices;
            const d1 = s1.data;
            const d2 = s2.data;
            const texts = s_texts.value;
            s_texts.value = "";
            const data = [];
            for (let i = 0; i < inds.length; i++) {
                data.push(" (Topic: " + d2['"""+topic_field+"""'][inds[i]] + ")")
                data.push("Document: " + d2['"""+document_field+"""'][inds[i]])
                data.push("\\r\\n")

            }
            s2.change.emit();
            s_texts.value = data.join("\\r\\n")
            s_texts.change.emit();




        """)
    )


    col1 = column(p1, multi_choice)
    col2 = column(data_table, selected_texts)
    col3 = column(p2)
    layout = row(col1, col2, col3)
    show(layout)


def create_labels(df, document_field):
    
    #Load Transformer Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Create Document Embeddings
    logging.info("Encoding Documents")
    doc_embeddings = model.encode(df[document_field])

    #Create UMAP Projection
    logging.info("Creating UMAP Projections")
    umap_proj = umap.UMAP(n_neighbors=50,
                              min_dist=0.01,
                              metric='correlation').fit_transform(doc_embeddings)

    #Create HDBScan Label
    logging.info("Finding Clusters with HDBScan")
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2).fit_predict(umap_proj)
    df["x"] = umap_proj[:,0]
    df["y"] = umap_proj[:,1]
    df["hdbscan_labels"] = hdbscan_labels

    return df

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


def create_tfidf(df, topic_data, document_field):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    lemma_docs = []
    for text in df[document_field].tolist():
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




def LeetTopic(df, document_field, html_filename, extra_fields=[], max_distance=.5):
    # df["documents"] = df[document_field]
    df = create_labels(df, document_field)
    logging.info("Calculating the Center of the Topic Clusters")
    topic_data = find_centers(df)
    logging.info(f"Recalculating clusters based on a max distance of {max_distance} from any topic vector")
    df = get_leet_labels(df, topic_data, max_distance)

    logging.info("Creating TF-IDF representation for documents")
    df, topic_data = create_tfidf(df, topic_data, document_field)

    logging.info("Creating TF-IDF representation for topics")
    df, topic_data = calculate_topic_relevance(df, topic_data)

    logging.info("Generating custom Bokeh application")
    create_html(df,
                document_field=document_field,
                topic_field="leet_labels",
                html_filename=html_filename,
                extra_fields=extra_fields)
    # create_html(df, "documents", topic_field, html_filename)
    return df, topic_data
























        #
