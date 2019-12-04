import tensorflow as tf
import keras
from keras.datasets import mnist

from matplotlib import pyplot as plot
import numpy as np


#Dataset provided from keras
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()


# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28) ),
    
    keras.layers.Dense(32, activation = tf.nn.relu ),
    keras.layers.BatchNormalization(),   
    
    keras.layers.Dense(32, activation = tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    

    keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.summary()


# We compile the model 
RMSProp_optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer = RMSProp_optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )


# Fit method return training history object including accuracy, loss, etc. 
training_history = model.fit(training_images, training_labels, epochs = 10)


# Model evaluation
test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
print("Final accuracy : {0:.2f}%".format(test_accuracy * 100) )


# Plot the training history (Optional)
plot.plot(training_history.history['acc'])
plot.xlabel('epoch')
plot.ylabel('accuracy')
plot.legend(['training'], loc= 'best')
plot.show()
