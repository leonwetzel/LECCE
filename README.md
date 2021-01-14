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
 
 You can easily download all auxiliary data by running the following commands
 in your Python console.
 
 ```python
>>> from lecce.information.corpus import Bible, Europarl, Pubmed
>>> corpora = [Bible(), Europarl(), Pubmed()]
>>> for corpus in corpora:
...     corpus.download()
```

Note that it can take some time to download all the data! We strongly
recommend indicating a file limit for Pubmed corpora. The default file
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
to word embeddings, which can be either Word2Vec or fastText-based.

### Word2Vec

In the example below,
  we train a new model based on a given list of tokenised sentences from our
  Pubmed corpus. The ```__init__``` function of ```Word2VecEmbedder```
can either create a new Word2Vec model (based on a given corpus) or load an
 existing model. It is also possible to use a base model, which is based on the
One Billion Word Benchmark ([Chelba et al., 2013](https://arxiv.org/abs/1312.3005)).

```python
>>> from lecce.feature.representation.word_embeddings import Word2VecEmbedder
>>> from lecce.feature.representation.generation import CorpusGenerator
>>> corpus = CorpusGenerator(files=["data/ez/base.txt", "data/ez/pubmed.txt"])
>>> embedder = Word2VecEmbedder(model_name=None, corpus=corpus, directory="embeddings")
```
We can easily load an existing model by using the following code, where we 
indicate the location of our base Word2Vec model.

```python
>>> from lecce.feature.representation.word_embeddings import Word2VecEmbedder
>>> embedder = Word2VecEmbedder(model_name="w2v_base.bin")
```

Word embeddings can be found in the directory ``embeddings`` per default. If
you have the word embeddings stored in a different directory, do not forget to
set the `directory` parameter when you instantiate a `Embedder` class object
(see the example below).

```python
>>> embedder = Word2VecEmbedder(model_name="w2v_base.bin", directory="my_embeddings")
```

Note that `Word2VecEmbedder` itself is not the model! It contains a `model` object
from Gensim, which can be used to transform text to word embeddings and to
calculate distance between tokens. `Embedder` class objects do contain a wrapper
function ``transform()`` for transforming tokens to their respective word embeddings.

```python
>>> embedder.transform("cheese")
>>> 
```

### fastText

#### Training corpus-specific word embeddings 

You can quickly train a new fastText model by using both the
``FastTextEmbedder`` and the `CorpusGenerator`. Thanks to `CorpusGenerator`, it
is possible to iterate rapidly over all the lines and sentences in the training
data. In the example below, you can see how the training of the bible embeddings
should work.

```python
>>> from lecce.feature.representation.word_embeddings import FastTextEmbedder
>>> from lecce.feature.representation.generation import CorpusGenerator
>>> sentences = CorpusGenerator(files=["data/ez/base.txt", "data/ez/bible.txt"])
>>> embedder = FastTextEmbedder(corpus=sentences)
```

## References

Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., & Robinson, T. (2013). One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005.

