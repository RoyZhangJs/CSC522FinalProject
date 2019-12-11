import preprocess
import numpy as np
import pandas as pd

df, probe_df = preprocess.clean_data()
# train_df, test_df = preprocess.split_train_test(df)

train_dict_user_id_to_index = {int(user_id): index for index, user_id in enumerate(df["user_id"].unique())}
train_dict_index_to_user_id = {index: int(user_id) for index, user_id in enumerate(df["user_id"].unique())}

train_dict_movie_id_to_index = {int(movie_id): index for index, movie_id in enumerate(df["movie_id"].unique())}
train_dict_index_to_movie_id = {index: int(movie_id) for index, movie_id in enumerate(df["movie_id"].unique())}


train_num_movies = len(df["movie_id"].unique())
train_num_users = len(df["user_id"].unique())
print("The number of movies in train data: ", train_num_movies)
print("The number of users in train data: ", train_num_users)


train_data = df.values
probe_data = probe_df.values
# test_data = test_df.values


train_matrix = np.zeros((train_num_movies, train_num_users), np.float16)
test_matrix = np.zeros((train_num_movies, train_num_users), np.float16)


for row in train_data:
    if row[2] != 0.0:
        i = train_dict_movie_id_to_index[int(row[0])]
        j = train_dict_user_id_to_index[int(row[1])]
        train_matrix[i, j] = row[2]

for row in probe_data:
    i = train_dict_movie_id_to_index[int(row[0])]
    j = train_dict_user_id_to_index[int(row[1])]
    test_matrix[i, j] = train_matrix[i, j]
    train_matrix[i, j] = 0.0

from sklearn.metrics.pairwise import pairwise_distances
item_similarity = pairwise_distances(train_matrix, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_matrix.T, item_similarity, type='item')
# user_prediction = predict(train_matrix, user_similarity, type='user')
print(item_prediction[0:8])
# print(user_prediction)
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(item_prediction, train_matrix.T)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_matrix.T)))