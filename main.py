
from keras.models import Sequential                        #Archytecture for our network
from keras.layers import Flatten, Dense                    #Layers for our neural network
from keras.datasets import mnist                           #handwritten images 28x28px
from keras.utils import normalize

import numpy as np



                                                             #28*28 images of hand-written digits 0-9
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()








xTrain = normalize(xTrain, axis=1)                           #normalize; values are scalled between 0 and 1
xTest = normalize(xTest, axis=1)

                                                                         #archytecture of the model
model = Sequential()
model.add(Flatten())                                                                   #Flatten is a layer that is build in keras
model.add(Dense(128, activation="relu"))                                                 #128 neurons
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))                                               #output layer; softmax for probability distribution

                                                                                                     #training of a model
                                                              #there are around 10 optimizers in keras; adam is the most basic one
                                                                                                       # #loss = degree of how much you got wrong; neural network doesnt try to optimise for acuraccy but it is always trying to minimize loss
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])                                               # #we track accuracy
model.fit(xTrain, yTrain,
          validation_data=(xTest, yTest),
          epochs=10,
          batch_size=100)                                                                  #training

#model.save("mnist.h5")
#print("Saving the model")
#show predictions for the first 3 images in the test set
#predictions = model.predict(xTest[:20])
#print(np.argmax(predictions, axis=1))
#print(yTest[:20])
                                                                                         #show actual results for the first 3 images in the test set






#Accuracy
#test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)

#print('\nTest accuracy:', test_acc)

score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


