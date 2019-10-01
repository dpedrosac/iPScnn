import talos as ta
from keras.models import Sequential
from keras.layers import Dense


def minimal():
    x, y = ta.templates.datasets.iris()

    p = {'activation': ['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'losses': ['logcosh'],
         'hidden_layers': [0, 1, 2],
         'batch_size': [20, 30, 40],
         'epochs': [10, 20]}

    def iris_model(x_train, y_train, x_val, y_val, params):

        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        model = Sequential()
        model.add(Dense(32, input_dim=4, activation=params['activation']))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=params['optimizer'], loss=params['losses'])

        out = model.fit(x_train, y_train, verbose=0,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=[x_val, y_val])

        return out, model

    scan_object = ta.Scan(x, y, model=iris_model, params=p)

    return scan_object