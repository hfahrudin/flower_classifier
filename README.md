# Flower-classification


## Introduction
Classify 5 kind of flowers which are daisy, tulip, rose, sunflower, and dandelion with convolutional neural network. I got the datasets from https://www.kaggle.com/alxmamaev/flowers-recognition.
I use [Keras](https://keras.io/) VGG16 as pre-trained model and deploying it in browser.

## How to deploy
1. First, you must have [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [Flask](http://flask.pocoo.org/).
2. Open a terminal in this folder.
3. run ```python app.y```.
4. Open your browser and go to ```http://0.0.0.0:5000/ ```.
5. The web page should look like [this](https://ibb.co/hBLSpK). Click ```choose file``` to input the image that you want to classify and ```predict``` button to display the result of model prediction.

## Make your own model!
1. First, download the [datasets](https://www.kaggle.com/alxmamaev/flowers-recognition).
2. Rename each ```.png``` to ```classname.index.jpg```. I already make the script to rename your file. Copy ```rename.py``` to each class folder and run it.
3. Split training set and test set to this kind folder structure :
```
/datasets
  /training_set
      /daisy
      /sunflower
      /tulip
      /rose
      /dandelion
  /test_set
      /daisy
      /sunflower
      /tulip
      /rose
      /dandelion
```
4. Open ```vgg16-model.py``` and modify the CNN.
5. run ```python vgg16-model.py``` in terminal. After the training process, it should appear ```model.json``` and ```model.h5```.
6. Move ```model.json``` and ```model.h5``` to ```model``` folder then replace the old file with new file.

## Contact me

if you have any question, email me for fast response.

email : fahrudinhasby12@gmail.com

facebook : https://www.facebook.com/hasby.fahrudin
