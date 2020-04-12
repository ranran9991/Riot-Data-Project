from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

class AutoEncoder():

    def __init__(self, input_dim, mid_size=2):
        input_df = Input(shape=(input_dim, ))
        self.mid_size = mid_size
        x = Dense(input_dim, activation='relu')(input_df)
        encoded = Dense(mid_size, activation='relu')(x)
        x = Dense(input_dim, activation='relu')(encoded)
        decoded = Dense(input_dim)(encoded)
        self.is_fit = False
        self.autoencoder = Model(input_df, decoded)
        self.encoder = Model(input_df, encoded)
        optimizer = optimizers.adam(learning_rate=0.01)
        self.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    def fit(self, X):
        self.is_fit = True
        self.autoencoder.fit(X, X,
         epochs=400,
         batch_size=20,
         shuffle=True,
         verbose=False
        )

    def predict(self, X):
        return self.encoder.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)