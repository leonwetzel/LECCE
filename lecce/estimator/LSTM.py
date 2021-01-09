import pandas
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM


def get_embedding(token, corpus):
    #pubmed_embeddings = ...
    #europarl_embeddings = ...
    #bible_embeddings = ...
    if corpus == 'bible':
        try:
            return bible_embeddings[word.lower()]
        except KeyError:
            return bible_embeddings['UNK'] # --> hier komt dan FastText??
    if corpus == 'biomed':
        try:
            return pubmed_embeddings[word.lower()]
        except KeyError:
            return pubmed_embeddings['UNK'] # --> hier komt dan FastText??
    if corpus == 'europarl':
        try:
            return europarl_embeddings[word.lower()]
        except KeyError:
            return europarl_embeddings['UNK'] # --> hier komt dan FastText??

#get data
X = []
Y = []
corpus = []
with open('lcp_single_train.tsv', 'r') as f:
    for line in f:
        split = line.strip().split('\t')
        X.append(get_embedding(split[3], split[1]))
        Y.append(split[4])
        corpus.append(split[1])

split_point = int(0.75*len(X))
X_train = X[:split_point]
Y_train = Y[:split_point]
X_test = X[split_point:]
Y_test = Y[split_point:] 
train_corpus = corpus[:split_point]
test_corpus = corpus[split_point:]

#build the model
model = Sequential()
model.add(Embedding(input_dim = ?, output_dim = ?, input_length = ?))
model.add(LSTM(units = 4, activation = 'adam', input_shape = ?))
model.add(Dense(input_dim = ?, units = 500, activation = 'adam'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 32, epochs = 100)

predictions = model.predict(X_test, batch_size = 5)
