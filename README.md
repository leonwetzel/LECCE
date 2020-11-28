# LECCE (LExiCal Complexity Estimator)
Code for the course Shared Task Information Science. Contains code of
LECCE (LExiCal Complexity Estimator), our submission for SemEval 2021 Task 1.


## Downloading and using data

Next to the LCP task data, we use additional data sources for training our
word embeddings. For example, we use the King James Version of the
 Bible for the bible-related word embeddings.
 
 You can easily download all auxilary data by running the following commands
 in your Python console.
 
 ```python
>>> from lecce.information.corpus import Bible, Europarl, Pubmed
>>> corpora = [Bible(), Europarl(), Pubmed()]
>>> for corpus in corpora:
...     corpus.download()
```

Note that it can take some time to download all the data!