from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

VERSION = '0.0.9'
DESCRIPTION = 'A new transformer-based topic modeling library.'

setup(
    name="leet_topic",
    author="WJB Mattingly",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["pandas>=1.0.0,<2.0.0",
                     "bokeh>=2.4.0, <2.4.3",
                     "sentence_transformers>=2.0.0",
                     "umap-learn>=0.5.0",
                     "hdbscan>=0.8.0",
                     "protobuf<=3.20.0",
                     "wrapt==1.14.0",
                     "tensorflow>=2.8.0",
                     "spacy>=3.3.0",
                     "gensim>=4.2.0",
                     ],
)
