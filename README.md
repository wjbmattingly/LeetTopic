![Leet Topic Logo](https://github.com/wjbmattingly/LeetTopic/raw/main/images/leettopic-logo.png)

LeetTopic builds upon Top2Vec, BerTopic and other transformer-based topic modeling Python libraries. Unlike BerTopic and Top2Vec, LeetTopic allows users to control the degree to which outliers are resolved into neighboring topics.

It also lets you turn any DataFrame into a Bokeh application for exploring your documents and topics.

# Installation

```python
pip install leet-topic
```

# Usage
```{python}
import pandas as pd
from leet_topic import leet_topic

df = pd.read_json("data/vol7.json")
new_df, topic_data = leet_topic.LeetTopic(df,
                                          document_field="descriptions",
                                          html_filename="demo.html",
                                          extra_fields=["names", "hdbscan_labels"],
                                          max_distance=.5)
```
