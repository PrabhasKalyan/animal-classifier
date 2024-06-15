from pydoc import classname
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#generates training data from directory using tf
train_generator = tf.keras.utils.image_dataset_from_directory(
 "animal_data",
 image_size=(224,224), #size of an image
 batch_size=128, #batch_size is the no.of images that are fed into the ram of the machine at a time
 seed=123, #using a fixed seed doesn't shuffle the image database
 shuffle=False
)


model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(224, 224, 3)),#convolutional layer with 3X3 filter
 tf.keras.layers.MaxPooling2D(2, 2),#maxpooling layer take max of 2X2 array
 tf.keras.layers.Conv2D(32, (3,3), activation='relu'),#convolutional layer ith 3X3 filter
 tf.keras.layers.MaxPooling2D(2,2),#maxpooling layer max of 2X2 array
 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),#convolutional layer with 3X3 filter
 tf.keras.layers.MaxPooling2D(2,2),#maxpooling layer max of 2X2 array
 tf.keras.layers.Flatten(),#flattens the 2d array to a vector to be fed into fully connected layer
 tf.keras.layers.Dense(512, activation='relu'),#fully connected layer with 512 parameters
 tf.keras.layers.Dense(15,activation='softmax')#Activation functions as there are 15 classes
])
model.compile(loss='sparse_categorical_crossentropy', #loss method is that is used especially for classification tasks
 optimizer='adam', #adam optimizer uses different learning rates for every step
 metrics=['accuracy'] ) #metrics used to evaluate the model is accuracy

history = model.fit(
 train_generator,
 steps_per_epoch=15, #how many batch gradient descent steps we want to have for one run of db
 epochs=15, #epochs is no.of times image dataset should be loaded
 verbose=1)


(images,label)=next(iter(train_generator))
batch_prediction=model.predict(images)
print(train_generator.class_names[np.argmax(batch_prediction[2])])
plt.imshow(np.array(images[2]).astype("uint8"))
plt.show()


