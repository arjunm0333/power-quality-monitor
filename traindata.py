import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

X = X.reshape(X.shape[0], X.shape[1], 1)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(1000,1)),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.2),

    LSTM(50),

    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=12, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

model.save("pq_model.h5")