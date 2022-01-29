import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

# Import data from CSV to pandas dataframe - use 
data = pd.read_csv("admissions_data.csv")

# Check dataframe
""" print(data.columns)
print(data.head(5)) """

# Select features
data_features = data.iloc[:,1:8]
""" print(data_features.columns)
print(data_features.head(5)) """

# Select labels
data_labels = data.iloc[:,8:9]
""" print(data_labels.columns) """

# Partition data to train/test sets
features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=0.2)
""" print(features_train.head(5)) """

# Initialize standard scaler
scaler = StandardScaler()

# Scale features
features_train = pd.DataFrame(scaler.fit_transform(features_train))
features_test = pd.DataFrame(scaler.transform(features_test))
""" print(features_train.head(5))
print(features_test.head(5)) """

# Model creation function
def create_model(feature_data):
    model = Sequential()
    num_features = feature_data.shape[1]
    input = tf.keras.Input(shape=(num_features))
    model.add(input)
    # Add 2 hidden layers and 2 dropout layers
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    # Add Adam optimizer and compile model
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

# Initialize sequential neural network model
admissions_model = create_model(data_features)

# Apply early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Fit model
history = admissions_model.fit(features_train, labels_train.to_numpy(), epochs=100, batch_size=4, verbose=1, validation_split=0.2, callbacks=[es])

# Print MSE, MAE
val_mse, val_mae = admissions_model.evaluate(features_test, labels_test, verbose=0)
print("MSE: " + str(val_mse) + "\nMAE: " + str(val_mae))

# Coefficient of determination
predicted_values = admissions_model.predict(features_test)
print(r2_score(labels_test, predicted_values))

# Plot MAE and residual MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')
 
# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')
 
# Used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('plots.png')
plt.show()