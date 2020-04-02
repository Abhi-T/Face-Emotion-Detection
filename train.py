from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#step 1 , building CNN
#initialize cnn
Classifier=Sequential()

#first conv layer and pooling
Classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,1), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

#second conv layer and pooling, input will be the first layer
Classifier.add(Convolution2D(32,(3,3), activation='relu'))
Classifier.add(MaxPooling2D(2,2))

#flattening the layer
Classifier.add(Flatten())

#adding a fully connected layer
Classifier.add(Dense(units=128, activation='relu'))
Classifier.add(Dense(units=3,activation='softmax')) #for more than 2 classification

#compiling the CNN
Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#step2 prepairing test-train data set
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('C:\\Users\\1000267332\\PycharmProjects\\OpenCV\\Practice\\Face-emotions\\Data\\train\\',
                                               target_size=(64,64), batch_size=5, color_mode='grayscale', class_mode='categorical')

test_set=test_datagen.flow_from_directory('C:\\Users\\1000267332\\PycharmProjects\\OpenCV\\Practice\\Face-emotions\\Data\\test\\',
                                               target_size=(64,64), batch_size=5, color_mode='grayscale', class_mode='categorical')

Classifier.fit_generator(training_set, steps_per_epoch=300, epochs=10,
                         validation_data=test_set, validation_steps=30)

#saving the model
model_json=Classifier.to_json()
with open('face_emotions-model.json','w') as json_file:
    json_file.write(model_json)
Classifier.save_weights('face_emotions_weights.h5')