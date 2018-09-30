# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras import optimizers
train_data_dir = 'dataset/training_set'
validation_data_dir = 'dataset/test_set'
## other
img_width, img_height = 299, 299
nb_train_samples = 100
nb_validation_samples = 800
top_epochs = 50
fit_epochs = 50
batch_size = 24
nb_classes = 5
nb_epoch = 10

#build CNN

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

input = Input(shape=(299,299, 3),name = 'image_input')

output_vgg16_conv = model_vgg16_conv(input)

for layer in model_vgg16_conv.layers[:15]:
    layer.trainable = False
model_vgg16_conv.summary()

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictions')(x)

vgg_model = Model(inputs=input, outputs=x)

vgg_model.summary()


#Image preprocessing and image augmentation with keras
vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)

# Setting learning data
train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

history = vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples
)

#Save the model
# serialize model to JSON
my_model_json = vgg_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(my_model_json)
# serialize weights to HDF5
vgg_model.save_weights("model.h5")
print("Saved model to disk")
