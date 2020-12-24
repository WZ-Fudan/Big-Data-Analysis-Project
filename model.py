# coding=utf-8

"""
@author: Zheng Wei
@date: 7/15/2020
"""

from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import *
from attention import Attention
from utils import *
from configuration import *


class NN:
    def __init__(self, embedding_weights, vocab_num):
        self.embedding = config['embedding']
        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.feature_dim = config['feature_dim']
        self.dropout_prob = config["dropout_prob"]
        self.filter_sizes = config["filter_size"]
        self.num_filters = config["num_filter"]
        self.batch_size = config["batch_size"]
        self.epoch = config["epoch"]
        self.model_num = config["model_num"]
        print(str(config))
        self.select_model(embedding_weights, vocab_num)

    def select_model(self, embedding_weights, vocab_num):
        if self.embedding != 1:
            input_shape = (self.sequence_length, self.embedding_dim)
        else:
            input_shape = (self.sequence_length,)

        model_input = Input(shape=input_shape)

        if self.embedding != 1:
            z = model_input
        else:
            z = Embedding(input_dim=vocab_num, output_dim=self.embedding_dim, input_length=self.sequence_length,
                          name="embedding")(model_input)
        print(z.shape)
        z = Dropout(self.dropout_prob[0])(z)

        if self.model_num == "CNN":
            # Convolutional block
            conv_blocks = []

            for sz in self.filter_sizes:
                conv = Convolution1D(filters=self.num_filters,
                                     kernel_size=sz,
                                     padding="valid",
                                     activation="relu",
                                     strides=1)(z)
                conv = MaxPooling1D(pool_size=2)(conv)
                conv = Flatten()(conv)
                conv_blocks.append(conv)
            z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
            z = Dense(self.embedding_dim, activation="tanh")(z)
            z = Dropout(self.dropout_prob[1])(z)
            z = Dense(self.feature_dim, activation="tanh")(z)

        elif self.model_num == "BiLSTM":
            # BiDirectional LSTM
            z = CuDNNLSTM(128, return_sequences=True)(z)
            avg_pool = GlobalAveragePooling1D()(z)
            max_pool = GlobalMaxPooling1D()(z)
            z = Concatenate()([avg_pool, max_pool])
            z = Dropout(self.dropout_prob[1])(z)
            # z = Attention(self.sequence_length)(z)
            z = Dense(self.feature_dim, activation="sigmoid")(z)

        else:
            raise Exception("Invalid model num!")

        self.model = Model(model_input, z)

        # Initialize weights with word2vec
        if self.embedding == 1:
            weights = np.array([v for v in embedding_weights.values()])
            print("Initializing embedding layer with word2vec weights, shape", weights.shape)
            embedding_layer = self.model.get_layer("embedding")
            embedding_layer.set_weights([weights])

    def train(self, X_vec, Y_vec):
        rmsprop = RMSprop(lr=0.001, rho=0.9)
        # model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=["mse"])
        self.model.compile(loss="mean_squared_error", optimizer=rmsprop, metrics=["mse"])
        self.history = self.model.fit(X_vec, Y_vec, batch_size=self.batch_size, epochs=self.epoch, verbose=0)
        return self.history

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)



