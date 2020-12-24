# coding=utf-8

"""
@author: Zheng Wei
@date: 7/15/2020
"""

import pickle
import numpy as np
import os
from gensim.models import word2vec
from os.path import join, exists, split
from configuration import *
from gensim.models import KeyedVectors


def save_pickle(data, name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    output = open(path + '/' + name + '.pkl', 'wb')
    pickle.dump(data, output)
    output.close()


def load_pickle(path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def find_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
    os.system('rm tmp')
    return str(np.argmax(memory_gpu))


def train_w2v(sentences, vocabulary, min_word_count=1, context=10):
    model_dir = os.getcwd() + '/w2vmodel/'
    # model_name = "{}_{:d}s_{:d}e_{:d}f".format(config["dataset"], config["sequence_length"], config["embedding_dim"],
    #                                            config["feature_dim"])
    model_name = "{:d}s_{:d}e_{:d}f".format(config["sequence_length"], config["embedding_dim"], config["feature_dim"])
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model.py \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model.py
        print('Training Word2Vec model.py...')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=config["embedding_dim"], min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model.py any further, calling
        # init_sims will make the model.py much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model.py for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model.py \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words word2vec_output_file = os.getcwd() + '/dataset/glove.6B/glove.6B.' + str(config[
    # "embedding_dim"]) + 'd.txt.word2vec' embedding_model = KeyedVectors.load_word2vec_format(word2vec_output_file,
    # binary=False)
    embedding_weights = {word: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for word in vocabulary}
    return embedding_weights


def eval_RMSE(U, V, user_id_map, movie_id_map, R):
    num = 0
    total = []
    for i in R.keys():
        i_rating = R[i]
        u = user_id_map[i]
        for j in i_rating.keys():
            m = movie_id_map[j]
            num += 1
            total.append((U[u].dot(V[m]) - i_rating[j]) ** 2)
    return np.sqrt(sum(total) / num)
