import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Input


def one_hot_encode(series):
    return pd.get_dummies(series).as_matrix()
    

def model_mnist():
    h= Input(shape=(784,))
    x =(Dense(64, activation='relu'))(h)
    y=(Dense(32, activation='relu')) (x)
    z=(Dense(16, activation='relu')) (y)
    res= (Dense(10, activation='softmax')) (z)
    return Model(inputs=h, outputs=res) 