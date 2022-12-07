![Leet Topic Logo](https://github.com/wjbmattingly/LeetTopic/raw/main/images/LeeTopic.png)

LeetTopic builds upon Top2Vec, BerTopic and other transformer-based topic modeling Python libraries. Unlike BerTopic and Top2Vec, LeetTopic allows users to control the degree to which outliers are resolved into neighboring topics.

It also lets you turn any DataFrame into a Bokeh application for exploring your documents and topics.

# Installation

```python
pip install leet-topic
```

# Parameters
df => a Pandas DataFrame that contains the documents that you want to model
document_field => the DataFrame column name where your documents sit
html_filename => the filename used to generate the Bokeh application
extra_fields => a list of extra columns to include in the Bokeh application
max_distance => The maximum distance between a document and the nearest topic vector to be considered for outliers

# Usage

```python
import pandas as pd
from leet_topic import leet_topic

df = pd.read_json("data/vol7.json")
new_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="descriptions",
                                          html_filename="demo.html",
                                          extra_fields=["names", "hdbscan_labels"],
                                          max_distance=.5)
```

This code above will generate a new DataFrame with the UMAP Projection (x, y), hdbscan_labels, and leet_labels, and top-n words for each document. It will also output data about each topic including the central plot of each vector, the documents assigned to it, top-n words associated with it.

Finally, the output will create an HTML file that is a self-contained Bokeh application like the image below.

![demo](https://github.com/wjbmattingly/LeetTopic/raw/main/images/leet-demo.png)

# Steps

LeetTopic takes an input DataFrame and converts the document field (texts to model) into embeddings. Next, UMAP is used to reduce the embeddings to 2 dimensions. HBDScan is then used to assign documents.

LeetTopic, like Top2Vec, then calculates the centroid for each topic based on the HBDScan labels while ignoring topic -1 (outlier). Next, all outliers are assigned to nearest topic centroid. Unlike Top2Vec, LeetTopic gives the user the ability to set a max distance so that outliers that are significantly away from a vector are not assigned to a nearest vector. At the same time, the output DataFrame contains information about the original HBDScan topics, meaning users know if a document was originally an outlier.

# Future Updates
A future update will allow users to control the parameters for UMAP and HBDScan. Another update will allow users to search the documents and topics via an Annoy Index.
