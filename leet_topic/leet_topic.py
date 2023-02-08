import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import math
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim.corpora as corpora
from random import random
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, DataTable, TableColumn, MultiChoice, HTMLTemplateFormatter, TextAreaInput, Div, TextInput
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
import string
import logging
import warnings
from annoy import AnnoyIndex

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





def create_html(df, document_field, topic_field, html_filename, topic_data, tf_idf, extra_fields=[], app_name=""):
    fields = ["x", "y", document_field, topic_field, "selected"]
    fields = fields+extra_fields
    output_file(html_filename)

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
    circle_kwargs = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s1,
                         "color": mapper
                        }
    scatter = p1.circle(**circle_kwargs)

    s2 = ColumnDataSource(data=dict(x=[], y=[], leet_labels=[]))
    p2 = figure(width=500, height=500, tools="pan,tap,lasso_select,wheel_zoom,box_zoom,box_select,reset", active_scroll="wheel_zoom", title="Analyze Selection", x_range=(df.x.min(), df.x.max()), y_range=(df.y.min(), df.y.max()))

    circle_kwargs2 = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s2,
                         "color": mapper
                        }
    scatter2 = p2.circle(**circle_kwargs2)

    multi_choice = MultiChoice(value=[], options=categories, width = 500, title='Selection:')
    data_table = DataTable(source=s2,
                           columns=columns,
                           width=700,
                           height=500,
                          sortable=True,
                          autosize_mode='none')
    selected_texts = TextAreaInput(value = "", title = "Selected texts", width = 700, height=500)
    top_search_results = TextAreaInput(value = "", title = "Search Results", width = 250, height=500)
    top_search = TextInput(title="Topic Search")
    doc_search_results = TextAreaInput(value = "", title = "Search Results", width = 250, height=500)
    doc_search = TextInput(title="Document Search")
    topic_desc = TextAreaInput(value = "", title = "Topic Descriptions", width = 500, height=500)
    
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


    multi_choice.js_on_change('value', CustomJS(args=dict(s1=s1, s2=s2, scatter=scatter, topic_desc=topic_desc, topic_data=topic_data), code="""
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
            let data = [];
            for (const [key, value] of topic_data) {
                for (let i=0; i < unchange_values.length; i++) {
                    if (key == unchange_values[i]) {
                        let keywords = value["key_words"];
                        data.push("Topic " + key + ": ");
                        for (let i=0; i < keywords.length; i++) {
                            data.push(keywords[i][0] + " " + keywords[i][1]);
                        }
                        data.push("\\r\\n");
                    }
                }
            }
            topic_desc.value = data.join("\\r\\n");
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
    
    top_search.js_on_change('value', CustomJS(args=dict(topic_data=topic_data, top_search_results=top_search_results, s4=multi_choice, s1=s1), code="""
        s1.selected.indices = []
        const search_term = cb_obj.value;
        let hits = [];
        let counter = 0;
        for (const [key, value] of topic_data) {
            const keywords = value["key_words"];
            for (let i=0; i < keywords.length; i++) {
                if (keywords[i][0] == search_term) {
                    hits.push([key, i]);
                }
            }
        }
        hits.sort(function(a, b) {
            return a[1] - b[1];
        });
        
        const data = [];
        if (hits.length) {
            for (let i = 0; i < hits.length; i++) { 
                data.push('Topic ' + hits[i][0] + ' has "' + search_term + '" as number ' + hits[i][1] + ' in its keyword list.');
                data.push("\\r\\n");
            }
        } else if (search_term != "") {
            data.push('No keyword matches with any topic for "' + search_term + '".');
        }
        
        top_search_results.value = data.join("\\r\\n");
        
        let inds = [];
        for (let i=0; i < hits.length; i++) {
            inds.push(hits[i][0]);
        }
        
        const res = [...new Set(inds)];
        
        s4.value = res.map(function(e){return e.toString()});
    
    """)
    )
    
    doc_search.js_on_change('value', CustomJS(args=dict(s1=s1, s2=s2, df=df.to_dict(), doc_search_results=doc_search_results, s4=multi_choice), code="""
        s1.selected.indices = []
        const search_term = cb_obj.value;
        let hits = [];
        let counter = 0;
        let id_count = 0;
        for (let i = 0; i < s1.data.top_words.length; i++) {
            for (let j = 0; j <s1.data.top_words[i].length; j++) { 
                if (search_term == s1.data.top_words[i][j][0]) { 
                    hits.push([id_count, j]);
                }
                
            }
            id_count = id_count + 1;
        }
        
        hits.sort(function(a, b) {
            return a[1] - b[1];
        });
        
        const data = [];
        if (hits.length) {
            for (let i = 0; i < hits.length; i++) { 
                data.push('Document ' + hits[i][0] + ' has "' + search_term + '" as number ' + hits[i][1] + ' in its top_words list.');
                data.push("\\r\\n");
            }
        } else if (search_term != "") {
            data.push('No keyword matches with any document for "' + search_term + '".');
        }
        
        doc_search_results.value = data.join("\\r\\n");
        
        let inds = [];
        for (let i=0; i <hits.length; i++) {
            inds.push(hits[i][0]);
            s1.selected.indices.push(hits[i][0]);
        }
        
        
        
        const d1 = s1.data;
        const d2 = s2.data;
        const d4 = s4;"""+list_creator(fields=fields, str_type="field")+
        """for (let i = 0; i < inds.length; i++) {"""+
        list_creator(fields=fields, str_type="push")+
        """}
        const res = [...new Set(d2['"""+topic_field+"""'])];
        s4.value = res.map(function(e){return e.toString()});
        s1.change.emit();
        s2.change.emit();
        
    
    """)
    )

    col1 = column(p1, multi_choice, topic_desc) 
    col2 = column(data_table, selected_texts)
    if tf_idf:
        col3 = column(p2, row(column(doc_search, doc_search_results), column(top_search, top_search_results)))
    else:
        col3 = column(p2)
    app_row = row(col1, col2, col3)
    if app_name != "":
        title = Div(text=f'<h1 style="text-align: center">{app_name}</h1>')
        layout = column(title, app_row, sizing_mode='scale_width')
    else:
        layout=app_row
    show(layout)


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
    lemma_docs = []
    for text in df[document_field].tolist():
        text = "".join([c for c in text if c.isdigit() == False])
        doc = nlp(text)
        lemma_docs.append(" ".join([token.lemma_.lower() for token in doc if token.text not in string.punctuation]))

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
    t = AnnoyIndex(doc_embeddings.shape[0], annoy_metric)
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
    spacy_modoel: str (Optional default en_core_web_sm)
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
























        #