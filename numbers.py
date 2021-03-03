"""
En este ejemplo, utilizaremos una libreria que esta ya dentro de tensorflow, con us net existente para entrenar info
"""

import tensorflow as tf
import numpy as np
import os

# para crear este projecto, vamos a usar keras, y su modelo sequencial.
# como solo tenemos 1 input (la imagen) y un output (el numero que es), es perfecto,
# pero no es bueno usar esto para otro tipo de cosas

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist # este es un modelo estandard que podemos usar (solo tiene numeros) 
from tensorflow.keras import utils
import matplotlib.pyplot as plt

# importamos keras

# se separa el dataset de mnist en 2 partes, el train, y el test (util despues)
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# podemos ver la data de forma rapida asi:
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

# este entrenamiento tiene valores de 0 a 255 en la escala de grises, y su valor de imagen es de 28 x 28 px
# dando un total de un array de 784 valores arreglados a una grafica con la forma presentada

# para optimizar el entrenamiento, en vez de hacer un rango de 0 a 255, tomamos un valor de 0 o 1, o bien, hay o no hay
# algun valor

# primero imprimimos la forma de todo para ver que en effecto cambie
print("X_train forma: ", X_train.shape)
print("y_train forma: ", y_train.shape)
print("X_test forma: ", X_test.shape)
print("y_test forma: ", y_test.shape)

# se construye un vector de los 28 x 28 pixeles
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784) # el 60000 y 10000 son la cantidad de datosque hay
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# y la hacemos equivalente a 1 o 0
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Matriz tipo tren de la info de entrenamiento", X_train.shape)
print("Matriz tipo tren de la info de pruebas", X_test.shape)

# el normalizar la informacion tambien ayuda a tener resultados mas limpios

# podemos ver que tambien tenemos el Y con valoresde 0 a 9 para comprobacion:
print(np.unique(y_train, return_counts=True))

# pero se puede optimizar cambiando los numeros por 1 y su pos
# estamos usando la transformacion proporcionada por keras
# a esto se la llama one-hot encoding https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
n_classes = 10
print("Forma antes de one-hot encoding: ", y_train.shape)
Y_train = utils.to_categorical(y_train, n_classes)
Y_test = utils.to_categorical(y_test, n_classes)
print("Forma despues de one-hot encoding: ", Y_train.shape)

# entonces llegamos al punto donde creamos el modelo, tenemos:
# un input tipo vector con 784 pixeles
# un area de procesamiento (hidden layer) y le vamos a poner 512 nodos
# un area de procesamiento (hidden layer) y le vamos a poner 512 nodos
# el output, que son 10 posibilidades distintas

# creamos el modelo:
model = keras.Sequential()

# le ponemos el primer layer de input (nota que el input se define dentro del param, pero tenemos nuestro primer layer de 512 nodos)
model.add(layers.Dense(512, input_shape=(784,)))
model.add(layers.Activation('relu'))                            
model.add(layers.Dropout(0.2))

# segundo:
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))

# y el output
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

# TODO ver que es el relu y softmax

# ahora podemos compilar nuestro modelo, para esto usamos model.Compile()

# le podemos dar lo que queremos obtener de el, el primer punto es loss, https://keras.io/api/losses/
# segundo punto son los meticos que queremos ver mientras se entrena,
# y el tercer punto es el optimizador, se usara adam, porque todos lo usan pero mas info en: https://arxiv.org/abs/1412.6980v8

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# y finalmente podemos entrenar a nuestro modelo:

# para entrenar un modelo, se usan varios paramentros, pero de los mas importantes son:
# batch_size - cuantos modelos se entrenan a la vez, mientras mas grande, mejor, pero mas recursos usa
# epochs - Cuantas generaciones de ajustes se usan? Mientras mas generaciones mas perfecto puede ser, pero llega un ounto donde solo ya no es necesario

history = model.fit(X_train, Y_train,
          batch_size=256, epochs=20,
          verbose=2,
          validation_data=(X_test, Y_test))

# podemos tambien guardar el modelo de la siguiente forma:
save_dir = "results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# y podemos ver las metricas en una grafica al terminar
'''
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
'''

# listo, ahora lo unico que falta es cargar el modelo y usarlo!
mnist_model = keras.models.load_model(os.path.join(save_dir, model_name)) # cargamos el modelo que creamos
predicted_classes = np.argmax(model.predict(X_test), axis=-1) # le damos lo que queremos predecir

# ver cuales predijimos de forma correcta y cuales no
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# y lo podemos ver con matplotlib
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect], 
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.show()