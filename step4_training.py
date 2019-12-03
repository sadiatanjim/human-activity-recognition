import numpy as np

fusedFrames = np.load("fusedFrames.npy")
Y = np.load("labels.npy")

from sklearn.preprocessing import StandardScaler

fusedFrames = fusedFrames.reshape(532, 306)

scaler = StandardScaler()
X = scaler.fit_transform(fusedFrames)
X = X.reshape(532, 102, 3)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, encoded_Y, test_size=0.2, random_state=42, stratify=encoded_Y
)

X_train = X_train.reshape(425, 102, 3, 1)
X_test = X_test.reshape(107, 102, 3, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(16, (2, 2), activation="relu", input_shape=X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation="relu"))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(6, activation="softmax"))

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    X_train, y_train, epochs=400, validation_data=(X_test, y_test), verbose=1
)

import matplotlib.pyplot as plt

def plot_learningCurve(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history["acc"])
    plt.plot(epoch_range, history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history["loss"])
    plt.plot(epoch_range, history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

print("Maximum validation accuracy: " + str(max(history.history['val_acc'])))

plot_learningCurve(history, 400)

