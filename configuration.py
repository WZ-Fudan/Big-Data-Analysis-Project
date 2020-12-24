# coding=utf-8

"""
@author: Zheng Wei
@date: 12/19/2019
"""

import os

config = {
    "embedding": 0,
    "model_num": "CNN",
    "sequence_length": 300,
    "embedding_dim": 256,
    "feature_dim": 50,
    "dropout_prob": [0.5, 0.8],
    "filter_size": [3, 4, 5],
    "num_filter": 100,
    "batch_size": 128,
    "epoch": 5,
    "max_epoch": 50,
    "lam_u": 100,
    "lam_v": 10,
    "dataset": '1m',
    "k-fold": "4-fold"
}

config_str = "{}_{:d}s_{:d}e_{:d}f_{:.2f}dp_{:d}n".format(config["dataset"] + config["k-fold"],
                                                          config["sequence_length"], config["embedding_dim"],
                                                          config["feature_dim"], config["dropout_prob"][1],
                                                          config["num_filter"])

prefix = str(config["sequence_length"]) + 's_' + str(config["embedding_dim"]) + 'e_'

load_dataset_path = os.getcwd() + '/dataset/' + config["dataset"] + '/'
save_dataset_path = load_dataset_path + config["k-fold"] + '/'

