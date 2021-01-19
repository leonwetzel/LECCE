from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.layers import LSTM, Bidirectional
from keras.callbacks import EarlyStopping

from lecce.feature.representation.word_embeddings import\
    FastTextEmbedder, Word2VecEmbedder
from definitions import ROOT_DIR

ft_bible = FastTextEmbedder(model_name="ft_bible.bin", directory=rf"{ROOT_DIR}/embeddings")
ft_europarl = FastTextEmbedder(model_name="ft_europarl.bin", directory=rf"{ROOT_DIR}/embeddings")
ft_pubmed = FastTextEmbedder(model_name="ft_pubmed.bin", directory=rf"{ROOT_DIR}/embeddings")

w2v_bible = Word2VecEmbedder(model_name="w2v_bible.bin", directory=rf"{ROOT_DIR}/embeddings")
w2v_europarl = Word2VecEmbedder(model_name="w2v_europarl.bin", directory=rf"{ROOT_DIR}/embeddings")
w2v_pubmed = Word2VecEmbedder(model_name="w2v_pubmed.bin", directory=rf"{ROOT_DIR}/embeddings")


def get_embedding(token, corpus, paradigm="ft"):
    """

    Parameters
    ----------
    token
    corpus
    paradigm

    Returns
    -------

    """
    token = token.lower()
    if paradigm.lower() == "ft":
        if corpus == 'bible':
            return ft_bible.transform(token)
        if corpus == 'pubmed':
            return ft_pubmed.transform(token)
        if corpus == 'europarl':
            return ft_europarl.transform(token)

    if paradigm.lower() == "w2v":
        if corpus == 'bible':
            return w2v_bible.transform(token)
        if corpus == 'pubmed':
            return w2v_pubmed.transform(token)
        if corpus == 'europarl':
            return w2v_europarl.transform(token)

#get data
X = []
Y = []
corpus = []
with open('lcp_single_train.tsv', 'r', encoding='utf-8') as f:
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

print(train_corpus)


#build the model
# model = Sequential()
# model.add(Embedding(input_dim=100, output_dim=100, input_length = ?))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dense(input_dim = ?, units = 500, activation = 'adam'))
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss', min_delta = 0, patience=10, mode='auto', verbose=1)
# model.fit(X_train, Y_train, batch_size = 32, epochs = 100, callbacks=[es])
#
# predictions = model.predict(X_test, batch_size = 5)
