# coding=utf-8

"""
@author: Zheng Wei
@date: 7/15/2020
"""

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold
import pandas as pd
from utils import *
from configuration import *


def save_data(train, valid, test, output_path, pre=''):
    R_u2m = []
    R_m2u = []
    for item in (train, valid, test):
        tmp_R_u2m = dict()
        for i in tuple(set(item["user_id"])):
            fixed_user = item[item["user_id"] == i]
            tmp_R_u2m[i] = dict(zip(fixed_user["movie_id"], fixed_user["rating"]))
        R_u2m.append(tmp_R_u2m)

        tmp_R_m2u = dict()
        for j in tuple(set(item["movie_id"])):
            fixed_movie = item[item["movie_id"] == j]
            tmp_R_m2u[j] = dict(zip(fixed_movie["user_id"], fixed_movie["rating"]))
        R_m2u.append(tmp_R_m2u)

    # save_pickle(self.plot2vector,
    #             name="movie_to_vector_" + str(self.embedding_dim) + 'd_' + str(self.sequence_length) + 'w',
    #             path=self.path)
    save_pickle(R_u2m[:2], name=pre + "TV_R_u2m", path=output_path)
    save_pickle(R_m2u[:2], name=pre + "TV_R_m2u", path=output_path)
    save_pickle(R_u2m[-1], name=pre + "test_R_u2m", path=output_path)
    save_pickle(R_m2u[-1], name=pre + "test_R_m2u", path=output_path)


class data_factory:
    def __init__(self, dim=config["embedding_dim"], length=config["sequence_length"]):
        self.plots = None
        self.users = None
        self.embedding_dim = dim
        self.sequence_length = length
        self.path = load_dataset_path
        # if os.path.isdir(path):
        #     self.path = path
        # else:
        #     raise Exception("Path doesn't exist!")

    def read_data(self):

        print("Loading User Data...")
        self.users = pd.read_csv(self.path + '/ml-' + config["dataset"] + '_ratings.dat', sep='::', header=None,
                                 engine='python')
        self.users.columns = ["user_id", "movie_id", "rating", "time"]
        del self.users["time"]

        print("Loading Plots Data...")
        self.plots = pd.read_csv(self.path + '/ml_plot.dat', sep='::', header=None, engine='python')
        self.plots.columns = ["movie_id", "plot"]
        # self.plots["plot"] = self.plots["plot"].apply(lambda x: ' '.join(x.split("|")[0].split("\t")[:-1]))
        self.plots["plot"] = self.plots["plot"].apply(
            lambda x: sum([item.split("\t")[:-1] for item in x.split("|")], []))

        movie_id_set = set(self.plots["movie_id"])
        self.users = self.users[self.users["movie_id"].isin(movie_id_set)]
        self.plots = self.plots[self.plots["movie_id"].isin(set(self.users["movie_id"]))]
        self.plots.reset_index(drop=True, inplace=True)

        bert_plots = []
        for item in self.plots["plot"].values:
            bert_plots.append(' '.join(item))
        save_pickle(bert_plots, name=prefix + "_bert_plots2matrix", path=load_dataset_path)
        # self.plots["plot"] = self.plots["plot"].apply(lambda x: x.split(' '))

        movie_ids = set(self.plots["movie_id"])
        user_ids = set(self.users["user_id"])
        self.movie_id_map = dict(zip(movie_ids, range(len(movie_ids))))
        self.user_id_map = dict(zip(user_ids, range(len(user_ids))))

    def padding_plots(self):
        print("Padding Plots...")
        self.plots2matrix = []
        self.vocabulary = {'<PAD>'}
        for p in self.plots["plot"]:
            self.vocabulary = self.vocabulary.union(set(p))
            l = len(p)
            if l >= self.sequence_length:
                self.plots2matrix.append(p[:self.sequence_length])
            else:
                self.plots2matrix.append(p + ['<PAD>'] * (self.sequence_length - l))
        # self.vocab_int_map = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        # self.int_vocab_map = dict(zip(range(len(self.vocabulary)), self.vocabulary))

        # for p in self.plots["plot"]:
        #     l = len(p)
        #     if l >= self.sequence_length:
        #         self.plots2matrix.append([self.vocab_int_map[word] for word in p[:self.sequence_length]])
        #     else:
        #         self.plots2matrix.append([self.vocab_int_map[word] for word in p]
        #                                  + [self.vocab_int_map['<PAD>']] * (self.sequence_length - l))
        # self.plots2matrix = np.arrayself.plots2matrix)
        save_pickle(self.plots2matrix, name=prefix + "_plots2matrix", path=load_dataset_path)
        save_pickle(self.vocabulary, name=prefix + "_vocabulary", path=load_dataset_path)
        # save_pickle(self.vocab_int_map, name="vocab_int_map", path=self.path)
        # save_pickle(self.int_vocab_map, name="int_vocab_map", path=self.path)

    def embedding_plots(self):
        print("Embedding Plots...")
        word2vec_output_file = self.path + '/glove.6B/glove.6B.' + str(self.embedding_dim) + 'd.txt.word2vec'
        if not os.path.isfile(word2vec_output_file):
            glove2word2vec(self.path + '/glove.6B/glove.6B.' + str(self.embedding_dim) + 'd.txt', word2vec_output_file)
        w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

        self.plots["plot"] = self.plots["plot"].apply(lambda x: list(filter(lambda i: i in w2v_model.vocab, x)))

        self.plot2vector = []
        for p in self.plots["plot"]:
            l = len(p)
            if l >= self.sequence_length:
                self.plot2vector.append(w2v_model[p][:self.sequence_length])
            else:
                self.plot2vector.append(np.vstack((w2v_model[p],
                                                   np.zeros((self.sequence_length - l, self.embedding_dim)))))

        self.plot2vector = np.array(self.plot2vector)

    def split_data(self, ratio=0.8, random_state=920):
        """
        Split randomly rating data into training set, valid set and test set with given ratio (valid+test)
        and save three data sets to given path.
        Note that the training set contains at least a rating on every user and item.
        """
        print("Split data and save data...")
        if self.users is None:
            self.read_data()
            self.padding_plots()

        save_pickle(self.movie_id_map, name="movie_id_map", path=load_dataset_path)
        save_pickle(self.user_id_map, name="user_id_map", path=load_dataset_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        for k, (train_index, test_index) in enumerate(kf.split(self.users)):
            train = self.users.iloc[train_index]
            assert len(set(train["user_id"])) == len(set(self.users["user_id"]))

            leaky_set = set(self.users["movie_id"]) - set(train["movie_id"])
            train = pd.concat([train, self.users[self.users["movie_id"].isin(leaky_set)]])

            no_train = self.users.drop(train.index)
            test = no_train.sample(frac=0.5, random_state=random_state)
            valid = no_train.drop(test.index)
            path = load_dataset_path + str(k) + '-fold/'
            print(path, load_dataset_path)
            if not os.path.exists(path):
                os.makedirs(path)
            save_data(train, valid, test, output_path=path)

    def ratio_split_data(self, ratio, random_state):
        """
        Split randomly rating data into training set, valid set and test set with given ratio (valid+test)
        and save three data sets to given path.
        Note that the training set contains at least a rating on every user and item.
        """
        train = self.users.sample(frac=ratio, random_state=random_state)
        leaky_set_user_id = set(self.users["user_id"]) - set(train["user_id"])
        leaky_set_movie_id = set(self.users["movie_id"]) - set(train["movie_id"])
        train = pd.concat([train, self.users[self.users["user_id"].isin(leaky_set_user_id)]])
        train = pd.concat([train, self.users[self.users["movie_id"].isin(leaky_set_movie_id)]])

        no_train = self.users.drop(train.index)
        test = no_train.sample(frac=0.5, random_state=random_state)
        valid = no_train.drop(test.index)

        save_data(train, valid, test, output_path=load_dataset_path + "r-fold/", pre=str(ratio))
