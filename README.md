![Leet Topic Logo](https://github.com/wjbmattingly/LeetTopic/raw/main/images/LeeTopic.png)

LeetTopic builds upon Top2Vec, BerTopic and other transformer-based topic modeling Python libraries. Unlike BerTopic and Top2Vec, LeetTopic allows users to control the degree to which outliers are resolved into neighboring topics.

It also lets you turn any DataFrame into a Bokeh application for exploring your documents and topics.

# Installation

```python
pip install leet-topic
```

# Parameters
- df => a Pandas DataFrame that contains the documents that you want to model
- document_field => the DataFrame column name where your documents sit
- html_filename => the filename used to generate the Bokeh application
- extra_fields => a list of extra columns to include in the Bokeh application
- max_distance => the maximum distance between a document and the nearest topic vector to be considered for outliers

# Usage

```python
import pandas as pd
from leet_topic import leet_topic

df = pd.read_json("data/vol7.json")
leet_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="descriptions",
                                          html_filename="demo.html",
                                          extra_fields=["names", "hdbscan_labels"],
                                          max_distance=.5)
```

## Multilingual Support
With LeetTopic, you can work with texts in any language supported by spaCy for lemmatization and any model from HuggingFace via Sentence Transformers.

Here is an examplpe working with Croatian

```python
import pandas as pd
from leet_topic import leet_topic

df = pd.DataFrame(["Bok. Kako ste?", "Drago mi je"]*20, columns=["text"])
leet_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="text",
                                          html_filename="demo.html",
                                          extra_fields=["hdbscan_labels"],
                                          spacy_model="hr_core_news_sm",
                                          max_distance=.5)
```

## Custom UMAP and HDBScan Parameters
It is often necessary to control how your embeddings are flattened with UMAP and clustered with HDBScan. As of 0.0.9, you can control these parameters with dictionaries.

```python
import pandas as pd
from leet_topic import leet_topic

df = pd.read_json("data/vol7.json")
leet_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="descriptions",
                                          html_filename="demo.html",
                                          extra_fields=["names", "hdbscan_labels"],
                                          umap_params={"n_neighbors": 15, "min_dist": 0.01, "metric": 'correlation'},
                                          hdbscan_params={"min_samples": 10, "min_cluster_size": 5},
                                          max_distance=.5)
```

# Outputs
This code above will generate a new DataFrame with the UMAP Projection (x, y), hdbscan_labels, and leet_labels, and top-n words for each document. It will also output data about each topic including the central plot of each vector, the documents assigned to it, top-n words associated with it.

Finally, the output will create an HTML file that is a self-contained Bokeh application like the image below.

![demo](https://github.com/wjbmattingly/LeetTopic/raw/main/images/leet-demo.png)

# Steps

LeetTopic takes an input DataFrame and converts the document field (texts to model) into embeddings. Next, UMAP is used to reduce the embeddings to 2 dimensions. HDBScan is then used to assign documents to topics. Like BerTopic and Top2Vec, at this stage, there are many outliers (-1).

LeetTopic, like Top2Vec, then calculates the centroid for each topic based on the HDBScan labels while ignoring topic -1 (outlier). Next, all outliers are assigned to nearest topic centroid. Unlike Top2Vec, LeetTopic gives the user the ability to set a max distance so that outliers that are significantly away from a vector are not assigned to a nearest vector. At the same time, the output DataFrame contains information about the original HDBScan topics, meaning users know if a document was originally an outlier.



# Future Roadmap
## 0.0.9
- Control UMAP parameters
- Control HDBScan parameters
- Multilingual support for lemmatization
- Multilingual support for embedding
- Add support for custom App Titles

## 0.0.10
- Output an Annoy Index so that the data can be queried

## 0.0.11
- Support for embedding text, images, or both via CLIP and displaying the results in the same bokeh application
