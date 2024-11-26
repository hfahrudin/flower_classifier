
<p align="center">
  <img width="750" height="375" src="https://user-images.githubusercontent.com/25025173/48326925-97100280-e66e-11e8-8fe1-4b9b4aa927f9.png">
</p>

## Introduction
Classify 5 kind of flowers which are daisy, tulip, rose, sunflower, and dandelion with convolutional neural network. I got the datasets from https://www.kaggle.com/alxmamaev/flowers-recognition.
I use [Keras](https://keras.io/) VGG16, Xception, Resnet50, and InceptionV3 as pre-trained model and deploying it in browser.

## How to deploy
1. First, you must have [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [Flask](http://flask.pocoo.org/).
2. Download weight and json file [here](http://www.mediafire.com/folder/k6ejyi8e3tie2,m1ohbjj6akbp7/shared).
3. Make weight and json folder in the root folder and put the weights and json files there. 
2. Open a terminal in this folder.
3. run ```python app.y```.
4. Open your browser and go to ```http://0.0.0.0:5000/ ```.
5. Click ```choose file``` to input the image that you want to classify and ```predict``` button to display the result of model prediction.

## Train your own model!
1. First, download the [datasets](https://www.kaggle.com/alxmamaev/flowers-recognition).
2. Split training set and test set to this kind folder structure :
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
4. Open ```{model name}_model.py```, you can choose the base model as you like, and edit the ```CNN``` part. 
5. run edited python file in terminal. After the training process, it should appear weight and json file named according to the base model.
6. Move them into the weight and json folder .

## Contact me

if you have any question, email me for fast response.

email : fahrudinhasby12@gmail.com

facebook : https://www.facebook.com/hasby.fahrudin
