from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
import pandas as pd

np.random.seed(123456)


def get_data():
    # Read the data and drop timestamp
    data = pd.read_csv('ratings.csv')
    data.drop('timestamp', axis=1, inplace=True)
    
    # Re-map the indices
    users = data.userId.unique()
    movies = data.movieId.unique()
    # Create maps from old to new indices
    moviemap={}
    for i in range(len(movies)):
        moviemap[movies[i]]=i
    usermap={}
    for i in range(len(users)):
        usermap[users[i]]=i
    
    # Change the indices
    data.movieId = data.movieId.apply(lambda x: moviemap[x])    
    data.userId = data.userId.apply(lambda x: usermap[x])    
        
    # Shuffle the data
    data = data.sample(frac=1.0).reset_index(drop=True)
    
    # Create a train/test split
    train, test = train_test_split(data, test_size=0.2)
    
    n_users = len(users)
    n_movies = len(movies)

    return train, test, n_users, n_movies


train, test, n_users, n_movies = get_data()

fts = 5

# Movie part. Input accepts the index as input
# and passes it to the Embedding layer. Finally,
# Flatten transforms Embedding's output to a
# one-dimensional tensor.
movie_in = Input(shape=[1], name="Movie")
mov_embed = Embedding(n_movies, fts, name="Movie_Embed")(movie_in)
flat_movie = Flatten(name="FlattenM")(mov_embed)

# Repeat for the user.
user_in = Input(shape=[1], name="User")
user_inuser_embed = Embedding(n_users, fts, name="User_Embed")(user_in)
flat_user = Flatten(name="FlattenU")(user_inuser_embed)

# Concatenate the Embedding layers and feed them 
# to the Dense part of the network
concat = Concatenate()([flat_movie, flat_user])
dense_1 = Dense(128)(concat)
dense_2 = Dense(32)(dense_1)
out = Dense(1)(dense_2)

# Create and compile the model
model = Model([user_in, movie_in], out)
model.compile('adam', 'mean_squared_error')

# Train the model on the train set
model.fit([train.userId, train.movieId], train.rating, epochs=10, verbose=1)

# Evaluate on the test set
print(metrics.mean_squared_error(test.rating, 
                                 model.predict([test.userId, test.movieId])))
