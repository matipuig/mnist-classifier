# MNIST classifier.

# Instalar la base de datos del MNIST.
from tensorflow import keras
from PIL import ImageTk, Image
from numpy import asarray, argmax
from keras.datasets import mnist
(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()

# Normalizar el DATASET.
X_train = X_train_raw.reshape(60000,784)
X_test = X_test_raw.reshape(10000, 784)
X_train = X_train / 255
X_test = X_test / 255

# Asignarle la forma vectorial.
Y_train = keras.utils.to_categorical(Y_train_raw, 10)
Y_test = keras.utils.to_categorical(Y_test_raw, 10)

# Generar la red neuronal.
from keras.layers.core import Dense, Dropout, Activation
model = keras.Sequential()
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compilar y entrenar la red neuronal.
print("Entrenando red neuronal...")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
result = model.fit(X_train, Y_train, batch_size=200, epochs=10, verbose=0, validation_data=(X_test, Y_test))
print("Red neuronal entrenada!")

def predict(image):
    # Abro la imagen con la librería Pillow.
    image = Image.open(image) 
    # La convierto a escala de grises.
    image = image.convert('L') 
    # La convierto al tamaño que necesita el modelo. 
    image = image.resize((28, 28))
    # La convierto a un array de numpy como neceista el modelo. 
    vector_image = asarray(image)

    # Obtengo la prediccion y al devuelvo.
    vector_image = vector_image.reshape(1, 784)
    result = argmax(model.predict(vector_image), axis=-1)[0]
    return result

def gethello():
    return "Working!"