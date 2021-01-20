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

### A note on file limits

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

### Merging text files

For our convenience, we concatenated corpus-related text files for easier
processing later on in the generation of the word embeddings. This applied to
both the Europarl and Pubmed corpora, which consist of multiple text files.

A simple bash command suffices for this operation to be replicated. For the
Europarl corpus, such a command would look like this:

```shell
foo@bar:~$ cat ep-*.txt > europarl.txt
```

We advise to store these concatenated text files separately from the original
text files. We stored these files in a new directory `ez` in the directory `data`.

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
array([-0.14763358, -0.6151625 , -4.5376935 , -1.3107201 ,  2.0699606 ,
       -3.2040992 ,  0.09065696,  0.7785856 ,  0.09909724,  0.21510065,
       -3.3662946 , -2.8873637 ,  5.504202  ,  1.5262611 ,  0.5984901 ,
       -5.0615873 ,  0.2658972 ,  0.41224727, -3.5005474 , -0.3451236 ,
       -0.31815568,  1.7148725 ,  0.5735119 ,  0.23066795,  2.3650339 ,
        1.2729089 , -2.4883304 ,  0.4951827 ,  3.40999   , -1.5533565 ,
        3.0736995 , -3.0521255 ,  2.2765718 , -1.7934179 , -0.3441617 ,
        1.5677184 , -0.10620257, -0.5721201 ,  0.27285793,  1.8695159 ,
       -0.13137296, -1.8967109 ,  1.3382695 ,  1.1660029 ,  5.9861336 ,
        2.8580027 , -0.2913719 , -3.192331  ,  0.72986996,  0.9690231 ,
        5.131245  , -1.9773573 ,  1.0814732 ,  2.8165162 ,  2.251758  ,
       -4.162695  , -2.4328437 , -2.08296   , -0.2894271 ,  1.3119276 ,
        0.8246829 ,  2.0581355 ,  2.41158   , -0.0378575 , -0.35984108,
       -1.4825187 , -1.7764221 , -0.3325412 , -0.71733415,  0.39039972,
       -1.8293386 ,  0.52562827,  0.80024123, -0.31259838,  0.39508483,
        1.9051372 , -1.8926342 ,  3.06406   , -2.234383  ,  1.3319718 ,
        0.8243203 ,  3.6725616 , -0.15421452,  4.2831674 , -0.10465561,
       -2.0435777 ,  1.0658191 , -0.87454593, -0.56031615, -3.2709572 ,
        1.9484522 ,  0.06203973,  0.39959732, -1.9495131 , -0.6922502 ,
        1.7539905 ,  2.261876  , -0.0587947 , -1.3472047 , -0.93357426],
      dtype=float32)
```

### fastText

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

In our examples, we always fit corpus-specific word embeddings with the base
corpus and the subcorpus. The base corpus allows the corpus-specific word
embeddings some generalisation, whilst maintaining .

## Running the system

Our regressor is based on the `LinearRegression` estimator of the Python
software package `scikit-learn`. This regressor takes into account several
features ($X$) to estimate the complexity score ($y$) of one or two tokens, present in
a given sentence.

## References

Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., & Robinson, T. (2013). One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005.

