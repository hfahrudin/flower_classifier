# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator


model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

model_vgg16_conv.summary()

from keras.layers import Input, Flatten, Dense
from keras.models import Model

input = Input(shape=(200,200, 3),name = 'image_input')

output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(5, activation='softmax', name='predictions')(x)

my_model = Model(inputs=input, outputs=x)

my_model.summary()


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classes = training_set.class_indices
print (classes)

my_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

my_model.fit_generator(training_set,
                         steps_per_epoch = 150,
                         epochs = 50,
                         validation_steps = 100,
                         validation_data = test_set)


#Save the model
# serialize model to JSON
my_model_json = my_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(my_model_json)
# serialize weights to HDF5
my_model.save_weights("model.h5")
print("Saved model to disk")
