from keras import layers, models, callbacks, optimizers
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

data_dim = 15
RNN = layers.GRU
DAYS = 120
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LAYERS = 0
EPOCHS = 200

# got minimum val_loss 72
# data_dim = 15
# RNN = layers.GRU
# DAYS = 120
# HIDDEN_SIZE = 128
# BATCH_SIZE = 64
# LAYERS = 0
# EPOCHS = 200
# with just dense(32), dense(1)

# with above and dense(128) after RNN, val_loss = 52

# with dense(128) before/after, loss != val_loss, overfitting

# with dense(32) before/after, loss =

# got minimum val_loss 204
# data_dim = 15
# RNN = layers.GRU
# DAYS = 120
# HIDDEN_SIZE = 32
# BATCH_SIZE = 32
# LAYERS = 0

def fillna(x):
    days = x.shape[1]
    dim = x.shape[2]

    x[x == -999999] = np.nan
    print('nan:', np.sum(np.isnan(x)))

    # forward fill
    for i in range(days-1):
        for j in range(dim):
            cur = x[:, i, j]
            next = x[:, i+1, j]
            where_nan = np.isnan(next)
            next[where_nan] = cur[where_nan]
    # backwards fill
    for i in range(days-1):
        for j in range(dim):
            cur = x[:, i+1, j]
            next = x[:, i, j]
            where_nan = np.isnan(next)
            next[where_nan] = cur[where_nan]

    print('nan:', np.sum(np.isnan(x)))

    # remaining values set to zero
    np.nan_to_num(x, copy=False)

    # todo: consider horizontal transfer/averages

def load_data():
    global x, y
    data = np.load('data2.npz')
    x = data['x']
    y = np.expand_dims(data['y'], axis=2)

    fillna(x)

    print('Data:', x.shape, y.shape)

    print('Zeros:', np.sum(x==0))
    print('-999999:', np.sum(x==-999999))
    print('nan:', np.sum(np.isnan(x)))

def train():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = models.Sequential()
    # returns a sequence of vectors
    #model.add(layers.Masking(mask_value=-999999, input_shape=(DAYS, data_dim)))
    model.add(layers.TimeDistributed(layers.Dense(32, activation="relu"), input_shape=(DAYS, data_dim)))
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    #model.add(layers.RepeatVector(DAYS))
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(32, activation="relu")))
    model.add(layers.TimeDistributed(layers.Dense(32, activation="relu")))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    # With GRU
    # Adam, loss = 1199

    # With LSTM
    # loss = 1217
    #optimizer = optimizers.RMSprop()
    # loss = 1217
    #optimizer = optimizers.RMSprop(lr=0.0001)
    # loss = 1209
    optimizer = optimizers.Adam()
    # GRU, loss = 1199
    #optimizer = optimizers.Adam(lr=0.0001)

    model.compile(loss='mse', optimizer=optimizer)
    model.summary()

    load_data()

    filepath = "models/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x, y,
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_split=0.05, callbacks=callbacks_list)

    model.save('models/model.h5')

def test():
    load_data()

    model = models.load_model('models/weights-improvement-118-10.36.h5')
    model.summary()

    global x, y
    n = x.shape[0]

    x = x[int(n*0.95):]
    y = y[int(n*0.95):]

    predictions = model.predict(x, batch_size=BATCH_SIZE)

    errors = (predictions-y).flatten()

    num_bins = 200

    fig, ax1 = plt.subplots()
    vals, bins, patches = ax1.hist(errors, num_bins, facecolor='blue', alpha=0.5)

    cdf = np.cumsum(vals/vals.sum())
    ax2 = ax1.twinx()
    ax2.plot(bins[:-1], cdf)

    fig.tight_layout()
    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()

