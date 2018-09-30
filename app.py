from flask import Flask, render_template, request
import numpy as np
import sys
import os
from skimage.transform import resize
from scipy.misc import imread
import base64
from keras.applications.xception import Xception
import re
sys.path.append(os.path.abspath('./model'))
from load import *


app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData1):
 imgstr = re.search(b'base64,(.*)',imgData1).group(1)
 with open('output.png','wb') as output:
     output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    img = imread('output.png')
    img = resize(img,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    with graph.as_default():
        prediction = model.predict(img)
        classes = np.argmax(prediction,axis=1)
        print (classes)
        if classes == 0:
            return "Daisy"
        elif classes == 1:
            return "Dandelion"
        elif classes == 2:
            return "rose"
        elif classes == 3:
            return "Sunflower"
        else :
            return "Tulip"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

