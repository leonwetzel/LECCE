#!/usr/bin/env python3
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

#def baseline_model():
    #model = Sequential()
    #model.add(Embedding)
    #model.add(Dense)
    #model.compile(loss="mean_squared_error", optimizer="adam")
    #model.fit(Xtrain, Ytrain, epochs=?, batch_size=?, verbose=1)
    #model.predict(Xtest, batch_size=?)
    #return model


def main():
    dataframe = pd.read_csv("lcp_single_train.tsv", sep="\t")
    dataframe = dataframe.drop("id", axis=1)
    X = dataframe
    y = dataframe.pop("complexity")
    europarl_embeddings = KeyedVectors.load_word2vec_format("../feature/representation/europarl_ft.bin",
                                                            binary=False, unicode_errors="ignore")
    bible_embeddings = KeyedVectors.load_word2vec_format("../feature/representation/bible_embeddings_W2V.bin",
                                                         binary=False)



if __name__ == "__main__":
    main()



