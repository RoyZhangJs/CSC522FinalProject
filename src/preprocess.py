import os
import natsort
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    training_dir = '../data/training_set/'
    data = list()
    for file in natsort.natsorted(os.listdir(training_dir)):
        file_name = os.path.join(training_dir, file)
        with open(file_name) as f:
            print("loading s%...", file_name)
            lines = f.readlines()
            movie_id = lines[0].strip()[:-1]
            for line in lines[1:]:
                line = line.split(",")
                new_line = [int(movie_id), int(line[0]), float(line[1])]
                data.append(new_line)
    np.save("data.npy", np.array(data))
    probe = list()
    with open("../data/probe.txt") as f:
        lines = f.readlines()
        movie_id = -1
        for line in lines:
            if ":" in line:
                movie_id = int(line.split(":")[0])
            else:
                new_line = [movie_id, int(line.strip())]
                probe.append(new_line)
    np.save("probe.npy", np.array(probe))


def clean_data():
    df = pd.DataFrame(np.load("data.npy"), columns=["movie_id", "user_id", "rating"])
    df_movie_summary = df.groupby("movie_id")["rating"].agg(["mean", "count"])
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.75), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_user_summary = df.groupby("user_id")["rating"].agg(["mean", "count"])
    df_user_summary.index = df_user_summary.index.map(int)
    user_benchmark = round(df_user_summary['count'].quantile(0.75), 0)
    drop_user_list = df_user_summary[df_user_summary['count'] < user_benchmark].index

    print('Customer minimum times of review: {}'.format(user_benchmark))

    print('Original Shape: {}'.format(df.shape))
    df = df[~df['movie_id'].isin(drop_movie_list)]
    df = df[~df['user_id'].isin(drop_user_list)]
    print('After Trim Shape: {}'.format(df.shape))

    probe_df = pd.DataFrame(np.load("probe.npy"), columns=["movie_id", "user_id"])
    probe_df = probe_df[~probe_df['movie_id'].isin(drop_movie_list)]
    probe_df = probe_df[~probe_df['user_id'].isin(drop_user_list)]
    return df, probe_df


def split_train_test(df):
    data = df.values
    train, test = train_test_split(data, test_size=0.2)
    print("Training set size: ", train.shape)
    print("Test set size: ", test.shape)
    return pd.DataFrame(train, columns=["movie_id", "user_id", "rating"]), \
           pd.DataFrame(test, columns=["movie_id", "user_id", "rating"])

