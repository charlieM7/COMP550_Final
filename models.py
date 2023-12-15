import pandas as pd
import tensorflow
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from keras.preprocessing import sequence, text
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.optimizers import Adam

def rnn_preprocess(X_train, X_test, Y_train, Y_test):
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    posts = pd.Series(X_train + X_test)
    length_of_the_messages = posts.str.split("\\s+")
    max_words = length_of_the_messages.str.len().max() + 1

    tokenizer = text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_seq_train = max(len(seq) for seq in X_train_seq)
    max_seq_test = max(len(seq) for seq in X_test_seq)

    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_seq_train)
    X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_seq_test)

    Y_train_cat = to_categorical(Y_train)
    Y_test_cat = to_categorical(Y_test)

    return max_words, X_train_pad, X_test_pad, Y_train_cat, Y_test_cat, max_seq_train, max_seq_test

def num_feature_normalization(train, test):
    feature_min = pd.concat([train, test], axis=0).min()
    feature_max = pd.concat([train, test], axis=0).max()
    feature_train_norm = (train - feature_min) / (feature_max - feature_min)
    feature_test_norm = (test - feature_min) / (feature_max - feature_min)

    return feature_train_norm, feature_test_norm

if __name__ == "__main__":
    cleaned_auto = pd.read_csv('data/cleaned_auto_labeled.csv')
    cleaned_hand = pd.read_csv('data/cleaned_hand_labeled.csv')
    cleaned_unlabeled = pd.read_csv('data/cleaned_unlabeled.csv')

    # text feature
    X_train, Y_train = cleaned_auto['Text'], cleaned_auto['Sentiment']
    X_test, Y_test = cleaned_hand['Text'], cleaned_hand['Sentiment']
    # num_comments feature
    C_train, C_test = cleaned_auto['Num_Comments'], cleaned_hand['Num_Comments']
    # score feature
    S_train, S_test = cleaned_auto['Score'], cleaned_hand['Score']

    # RNN
    max_words, X_train_pad, X_test_pad, Y_train_cat, Y_test_cat, max_seq_train, max_seq_test = \
        rnn_preprocess(X_train, X_test, Y_train, Y_test)
    C_train_norm, C_test_norm = num_feature_normalization(C_train, C_test)
    S_train_norm, S_test_norm = num_feature_normalization(S_train, S_test)

    X_train_rnn, X_val_rnn, C_train_rnn, C_val_rnn, S_train_rnn, S_val_rnn, Y_train_rnn, Y_val_rnn = \
        train_test_split(X_train_pad, C_train_norm, S_train_norm, Y_train_cat, test_size=0.2, random_state=42)

    embedding_dim = 3
    lstm_units = 128

    # 1 feature rnn
    text_input = Input(shape=(max_seq_train,))
    embedding_layer = Embedding(max_words, embedding_dim)(text_input)
    lstm_layer = LSTM(lstm_units)(embedding_layer)
    text_output = Dense(1, activation='linear')(lstm_layer)

    rnn_model = Model(inputs=text_input, outputs=text_output)

    rnn_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    rnn_model.fit(X_train_rnn, Y_train_rnn, epochs=10, batch_size=32, validation_data=(X_val_rnn, Y_val_rnn), verbose=1)

    rnn_model_results = rnn_model.evaluate(X_test_pad, Y_test_cat)
    print("Text-Only Model:")
    print("Mean Squared Error:", rnn_model_results[0])
    print("Mean Absolute Error:", rnn_model_results[1])
    print("\n")

    # 3 features rnn
    comments_input = Input(shape=(1,))
    comments_dense = Dense(32, activation='relu')(comments_input)
    score_input = Input(shape=(1,))
    score_dense = Dense(32, activation='relu')(comments_input)

    merged = concatenate([lstm_layer, comments_dense, score_dense])
    merged_dense = Dense(32, activation='relu')(merged)
    merged_output = Dense(1, activation='linear')(merged_dense)
    rnn_model_ft = Model(inputs=[text_input, comments_input, score_input], outputs=merged_output)

    rnn_model_ft.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    rnn_model_ft.fit([X_train, C_train_rnn, S_train_rnn], Y_train_rnn, epochs=10, batch_size=32,
                     validation_data=([X_val_rnn, C_val_rnn, S_val_rnn], Y_val_rnn), verbose=1)

    rnn_model_ft_results = rnn_model_ft.evaluate([X_test_pad, C_test_norm, S_test_norm], Y_test_cat)
    print("Text-Only Model:")
    print("Mean Squared Error:", rnn_model_ft_results[0])
    print("Mean Absolute Error:", rnn_model_ft_results[1])
    print("\n")

    # vectorizer = TfidfVectorizer()
    # X_train_vect = vectorizer.fit_transform(X_train)
    # vec_X_test = vectorizer.transform(X_test)
    # vec_X_train_ft_vect = vectorizer.fit_transform(X_train_ft)
    # vec_X_test_ft = vectorizer.transform(X_test_ft)


