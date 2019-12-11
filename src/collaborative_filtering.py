import preprocess
import numpy as np
import math
import matplotlib.pyplot as plt

num_iterations = 100
num_features = 100
alpha = 0.01
llambda = 1.5

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

# test_num_movies = len(df["movie_id"].unique())
# test_num_users = len(df["user_id"].unique())
# print("The number of movies in test data: ", test_num_movies)
# print("The number of users in test data: ", test_num_users)

train_data = df.values
probe_data = probe_df.values
# test_data = test_df.values

X = np.random.uniform(-0.2, 0.2, (num_features, train_num_movies))
Theta = np.random.uniform(-0.2, 0.2, (num_features, train_num_users))
Y = np.zeros((train_num_movies, train_num_users), np.float16)
R = np.zeros((train_num_movies, train_num_users), np.float16)

for row in train_data:
    if row[2] != 0.0:
        i = train_dict_movie_id_to_index[int(row[0])]
        j = train_dict_user_id_to_index[int(row[1])]
        Y[i, j] = row[2]
        R[i, j] = 1.0

test_targets= []
for row in probe_data:
    movie_index = train_dict_movie_id_to_index[row[0]]
    user_index = train_dict_user_id_to_index[row[1]]
    test_targets.append(Y[movie_index, user_index])
    Y[movie_index, user_index] = 0.0
    R[movie_index, user_index] = 0.0


Y = Y - np.mean(Y, axis=1, keepdims=True)               # mean normalization over user ratings
Lambda = np.ones((num_features, 1)) * llambda
params = {"X": X, "Theta": Theta, "loss": math.inf}
lr_adjustments = 0

train_MRSE = list()
test_MRSE = list()

for num_iteration in range(num_iterations):
    X_old, Theta_old = X.copy(), Theta.copy()
    Y_hat = (np.dot(X_old.T, Theta_old) - Y) * R

    X = X_old - alpha * (np.dot(Theta_old, Y_hat.T) + Lambda * X_old)
    Theta = Theta_old - alpha * (np.dot(X_old, Y_hat) + Lambda * Theta_old)

    loss = 0.5 * np.sum(((np.dot(X.T, Theta) - Y) ** 2) * R) + \
           0.5 * llambda * np.sum(X ** 2) + 0.5 * llambda * np.sum(Theta ** 2)

    if loss / params["loss"] > 1.0:
        alpha /= 2
        X = X_old.copy()
        Theta = Theta_old.copy()
        print("num_iteration: {} loss: {} --> adjusting learning rate".format(num_iteration + 1 - lr_adjustments, loss))
        lr_adjustments += 1
    else:
        params["X"] = X
        params["Theta"] = Theta
        params["loss"] = loss
        print("num_iteration: {} loss: {} lr: {}".format(num_iteration + 1 - lr_adjustments, loss, alpha))

    # targets = []
    # predictions = []
    # for i in range(0, train_num_movies):
    #     for j in range(0, train_num_users):
    #         if R[i, j] != 0:
    #             predictions.append(np.dot(X.T[i], Theta.T[j].T))
    #             targets.append(Y[i, j])
    # print("Train RSME for %dth iteration: %f" % (num_iteration + 1, np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())))

    test_predictions = []

    for row in probe_data:
        movie_index = train_dict_movie_id_to_index[row[0]]
        user_index = train_dict_user_id_to_index[row[1]]
        test_predictions.append(np.dot(X.T[movie_index], Theta.T[user_index].T))
    mrse = np.sqrt(((np.array(test_predictions) - np.array(test_targets)) ** 2).mean())

    print("Probe RSME for %dth iteration: %f" % (num_iteration + 1 - lr_adjustments, mrse))

    test_MRSE.append(mrse)


x2 = np.arange(0, len(test_MRSE), 1)
plt.title("Test MRSE")
plt.xlabel("epoch")
plt.ylabel("MRSE")
plt.plot(x2, test_MRSE)
plt.savefig("Test_MRSE.png")