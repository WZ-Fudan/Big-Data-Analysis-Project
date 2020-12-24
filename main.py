# coding=utf-8

"""
@author: Zheng Wei
@date: 7/15/2020
"""

from data_preprocessing import *
from utils import *
from model import NN
from configuration import *
from keras_bert import extract_embeddings
import time
import os


def generate_data(need_split=False, random_state=920):
    df = data_factory()
    df.read_data()
    df.padding_plots()
    # df.embedding_plots()
    if need_split:
        df.split_data(random_state=random_state)


def generate_ratio_data(random_state=920):
    df = data_factory()
    df.read_data()
    for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        df.ratio_split_data(ratio, random_state=random_state)


def train_model(seed=920):
    np.random.seed(seed)

    embedding = config["embedding"]
    feature_dim = config["feature_dim"]

    plots_matrix = load_pickle(load_dataset_path + prefix + '_plots2matrix.pkl')
    vocabulary = load_pickle(load_dataset_path + prefix + '_vocabulary.pkl')
    embedding_weights = train_w2v(sentences=plots_matrix, vocabulary=vocabulary)

    if embedding == 0:
        matrix = np.array([[embedding_weights[w] for w in p] for p in plots_matrix])
        print(matrix.shape)
    elif embedding == 1:
        vocab2int = dict(zip(vocabulary, range(len(vocabulary))))
        matrix = np.array([[vocab2int[w] for w in p] for p in plots_matrix])
    else:
        plots_matrix = load_pickle(load_dataset_path + prefix + '_bert_plots2matrix.pkl')
        model_path = './' + str(config["embedding_dim"]) + '/'
        print("Bert Model...")
        # texts = ['all work and no play', 'makes jack a dull boy~']
        matrix = extract_embeddings(model_path, plots_matrix)
        modified_matrix = []
        for item in matrix:
            np_array = np.random.uniform(-0.25, 0.25, (config["sequence_length"], config["embedding_dim"]))
            l = min(config["sequence_length"], len(item))
            np_array[:l] = item[:l]
            modified_matrix.append(np_array)
        matrix = np.array(modified_matrix)
        print("Done!")
        print(matrix.shape)

    TV_R_u2m = load_pickle(save_dataset_path + '/TV_R_u2m.pkl')
    TV_R_m2u = load_pickle(save_dataset_path + '/TV_R_m2u.pkl')
    test_R_u2m = load_pickle(save_dataset_path + '/test_R_u2m.pkl')

    # ratio = 0.7
    # print("Ratio: ", ratio)
    # TV_R_u2m = load_pickle(load_dataset_path + "r-fold/" + str(ratio) + 'TV_R_u2m.pkl')
    # TV_R_m2u = load_pickle(load_dataset_path + "r-fold/" + str(ratio) + 'TV_R_m2u.pkl')
    # test_R_u2m = load_pickle(load_dataset_path + "r-fold/" + str(ratio) + 'test_R_u2m.pkl')

    movie_id_map = load_pickle(load_dataset_path + '/movie_id_map.pkl')
    user_id_map = load_pickle(load_dataset_path + '/user_id_map.pkl')
    # vectors = load_pickle(path + '/dataset/movie_to_vector_' + str(embedding_dim) +
    #                       'd_' + str(sequence_len) + 'w' + '.pkl')

    max_epoch = config["max_epoch"]
    lam_u = config["lam_u"]
    lam_v = config["lam_v"]

    train_R_u2m = TV_R_u2m[0]
    train_R_m2u = TV_R_m2u[0]
    valid_R_u2m = TV_R_u2m[1]
    # valid_R_m2u = TV_R_m2u[1]

    num_users = len(train_R_u2m)
    num_movies = len(train_R_m2u)
    U = np.random.uniform(size=(num_users, feature_dim))
    V = np.random.uniform(size=(num_movies, feature_dim))
    CNN_model = NN(embedding_weights, len(vocabulary))
    from keras.utils import plot_model
    plot_model(CNN_model.model, to_file='model4.png', show_layer_names=False)
    raise Exception("Fuck!")

    Cnn = CNN_model.predict(matrix)



    print("Training Process...")
    # pre = str(ratio)
    f1 = open(os.getcwd() + "/logs/total_state_.log", 'a+')
    f = open(os.getcwd() + "/logs/state_" + config_str + '.log', 'w')

    stop_early_monitor = [2] * 5
    for epoch in range(max_epoch):
        t1 = time.time()
        print("Epoch ", epoch, end=': ')
        loss = 0

        # Update user vector
        for i in train_R_u2m.keys():
            Vi = V[[movie_id_map[k] for k in (train_R_u2m[i].keys())]]
            A = lam_u * np.eye(feature_dim, feature_dim) + Vi.T.dot(Vi)
            B = (Vi * np.tile(list(train_R_u2m[i].values()), (feature_dim, 1)).T).sum(0)
            U[user_id_map[i]] = np.linalg.solve(A, B)
            loss += lam_u / 2 * U[user_id_map[i]].dot(U[user_id_map[i]])

        # Update movie vector
        for j in train_R_m2u.keys():
            Ui = U[[user_id_map[k] for k in train_R_m2u[j].keys()]]
            A = lam_v * np.eye(feature_dim, feature_dim) + Ui.T.dot(Ui)
            B = (Ui * np.tile(list(train_R_m2u[j].values()), (feature_dim, 1)).T).sum(0) \
                + lam_v * Cnn[movie_id_map[j]]
            V[movie_id_map[j]] = np.linalg.solve(A, B)
            loss += 0.5 * (np.array(list(train_R_m2u[j].values())) ** 2).sum()
            loss += 0.5 * np.dot(V[movie_id_map[j]].dot(Ui.T.dot(Ui)), V[movie_id_map[j]])
            loss -= np.sum((Ui.dot(V[movie_id_map[j]])) * np.array(list(train_R_m2u[j].values())))

        history = CNN_model.train(matrix, V)

        # retrain CNN vector
        Cnn = CNN_model.predict(matrix)
        cnn_loss = history.history['loss'][-1]

        # calculate the loss
        loss += 0.5 * lam_v * cnn_loss * num_movies

        # compute the RMSE for three datasets
        tr_eval = eval_RMSE(U, V, user_id_map, movie_id_map, train_R_u2m)
        val_eval = eval_RMSE(U, V, user_id_map, movie_id_map, valid_R_u2m)
        test_eval = eval_RMSE(U, V, user_id_map, movie_id_map, test_R_u2m)
        t2 = time.time()

        stop_early_monitor.append(val_eval)
        converge = (np.abs(np.array(stop_early_monitor[-5:]) - np.array(stop_early_monitor[-6:-1]))).mean()

        print("Time: {:.1f}s, Loss: {:.4f}, Converge: {:.5f}, Training RMSE: {:.4f}, Valid RMSE: {:.4f}, "
              "Test RMSE: {:.4f}".format(t2 - t1, loss, converge, tr_eval, val_eval, test_eval))
        f.write("Epoch: {}, Time: {:.1f}, Loss: {:.4f}, Converge: {:.5f}, Training RMSE: {:.4f}, Valid RMSE: {:.4f}"
                ", Test RMSE: {:.4f}\n".format(epoch, t2 - t1, loss, converge, tr_eval, val_eval, test_eval))
        if converge < 1e-4:
            f1.write(config_str + "Epoch: {}, Training RMSE: {:.4f}, Valid RMSE: {:.4f}, Test RMSE: {:.4f}\n".
                     format(epoch, tr_eval, val_eval, test_eval))
            break

    # save model.py and result
    CNN_model.save_model(model_path=os.getcwd() + '/result/' + config_str + 'cnn_weights.h5')
    save_pickle(data=U, name=config_str + "_U", path=os.getcwd() + '/result/')
    save_pickle(data=V, name=config_str + "V", path=os.getcwd() + '/result/')
    save_pickle(data=Cnn, name=config_str + "Cnn", path=os.getcwd() + '/result/')

    f.close()
    f1.close()


if __name__ == "__main__":
    gpu_num = find_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    print("Using the gpu: ", gpu_num)
    # generate_ratio_data()
    # generate_data(need_split=False, random_state=920)
    train_model()
    # train_model(CNN_model, os.getcwd(), cnn_configuration)
