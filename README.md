# LECCE (LExiCal Complexity Estimator)

![LECCE](https://github.com/leonwetzel/LECCE/workflows/LECCE/badge.svg?branch=master)

Code for the course Shared Task Information Science. Contains code of
LECCE (LExiCal Complexity Estimator), our submission for SemEval 2021 Task 1.

## Downloading and using data

Next to the LCP task data, we use additional data sources for training our
word embeddings. For example, we use the King James Version of the
 Bible for the bible-related word embeddings. We also use English proceedings
 from the European Parliament and abstracts from Pubmed articles to train
 word embeddings for their respective corpus type.
 
 You can easily download all auxilary data by running the following commands
 in your Python console.
 
 ```python
>>> from lecce.information.corpus import Bible, Europarl, Pubmed
>>> corpora = [Bible(), Europarl(), Pubmed()]
>>> for corpus in corpora:
...     corpus.download()
```

Note that it can take some time to download all the data! We strongly
recommend to indicate a file limit for Pubmed corpora. The default file
limit is 200. Depending on your amount of disk space, we advise to
maintain a modest file limit. As the Pubmed files are downloaded
from the website of the National Center for Biotechnology Information
 via FTP, we are not really in control of the total amount of files on
 their server.

```python
>>> pm = Pubmed(file_limit=200)
>>> pm.download()
```

The data size can be considerable, even after extracting the article abstracts!

## Converting text to word embeddings

LECCE contains functionality to convert text from our various sources
to word embeddings, which can be either word2vec or fasttext-based.

### word2vec

In the example below,
  we train a new model based on a given list of tokenised sentences from our
  Pubmed corpus. The ```__init__``` function of ```Word2vecEmbedder```
can either create a new word2vec model (based on a given corpus) or load an
 existing model. It is also possible to use the GoogleNews-vectors model, which
 is particularly suitable for development work and testing.

```python
>>> from lecce.feature.representation.word_embeddings import Word2VecEmbedder
>>> corpus = pm.to_list_of_sentences("data/pubmed")
>>> embedder = Word2VecEmbedder(model_name=None, corpus=corpus, directory="embeddings")
```
We can easily load an existing model by using the following code, where we 
indicate the location of our custom model.

```python
>>> from lecce.feature.representation.word_embeddings import Word2VecEmbedder
>>> embedder = Word2VecEmbedder(model_name="my_cool_w2v_model.bin")

```

Note that `Word2VecEmbedder` itself is not the model! It contains a `model` object
from Gensim, which can be used to transform text to word embeddings and to
calculate distance between tokens.

### Fasttext

