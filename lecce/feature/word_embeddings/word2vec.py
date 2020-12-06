from gensim.models import Word2Vec


class Embedding:
    def __init__(self, model_name, corpus):
        """

        Parameters
        ----------
        model_name
        corpus
        """
        if model_name and not corpus:
            model = Word2Vec.load(model_name)
        elif corpus:
            model = Word2Vec(sentences=corpus, size=300,
                             alpha=0.025, window=5, min_count=5,
                             max_vocab_size=None, sample=0.001, seed=1,
                             workers=3, min_alpha=0.0001, sg=0, hs=0,
                             negative=5, ns_exponent=0.75, cbow_mean=1,
                             iter=5, null_word=0, trim_rule=None,
                             sorted_vocab=1, batch_words=10000,
                             compute_loss=False, callbacks=(),
                             max_final_vocab=None)
            model_name = input("Please enter the filename for your new"
                               "word2vec model: ")
        else:
            raise ValueError("Cannot create empty Embedding object!"
                             " Please refer to an existing model by"
                             " providing the filename of the model OR"
                             " provide a list of sentences for the"
                             " creation of a new model.")

        model.save(str(model_name))

    def transform(self):
        pass
