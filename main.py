from keras.models import Sequential        #Archytecture for our network
from keras.layers import Flatten, Dense    #Layers for our neural network
from keras.datasets import mnist           #handwritten images 28x28px
from keras.utils import normalize
import numpy as np

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()   #28*28 images of hand-written digits 0-9

xTrain = normalize(xTrain, axis=1)                #normalize; values are scalled between 0 and 1                       
xTest = normalize(xTest, axis=1)

                                                                       
model = Sequential()                              #archytecture of the model
model.add(Flatten())                              #Flatten is a layer that is build in keras; it flattens data
model.add(Dense(128, activation="relu"))          #128 neurons                                      
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))        #output layer; softmax for probability distribution

#compiling                                                                                                   
model.compile(optimizer="adam",                           #there are around 10 optimizers in keras; adam is the most basic one     
              loss="sparse_categorical_crossentropy",     #loss = degree of how much you got wrong; neural network doesnt try to optimise for acuraccy but it is always trying to minimize loss
              metrics=["accuracy"])                       #tracks accuracy                                                   
#training                                                                                                       
model.fit(xTrain, yTrain,
          validation_data=(xTest, yTest),
          epochs=10,
          batch_size=100)                                                                  


#show predictions for the first 20 images in the test set
predictions = model.predict(xTest[:20])
print(np.argmax(predictions, axis=1))
print(yTest[:20])                                             #show actual results for the first 20 images in the test set
                                                                                        











