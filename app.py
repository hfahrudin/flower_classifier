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
import json

app = Flask(__name__)

global model, graph
allmodel, graph = init()
model_name = ['resnet50', 'vgg16', 'xception', 'inceptionv3']

def convertImage(imgData1):
 imgstr = re.search(b'base64,(.*)',imgData1).group(1)
 with open('output.png','wb') as output:
     output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    jsonmap = []
    imgData = request.get_data()
    convertImage(imgData)
    img = imread('output.png')
    img = resize(img,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    with graph.as_default():
        for i in range(0, len(allmodel)):      
            prediction = allmodel[i].predict(img)
            key = {
                    'daisy' : str(prediction[0][0]),
                    'dandelion' : str(prediction[0][1]),
                    'rose' : str(prediction[0][2]),
                    'sunflower' : str(prediction[0][3]),
                    'tulip' : str(prediction[0][4]),
                    'model' : model_name[i]
                    }
            jsonmap.append(key)
        return json.dumps({'a' : jsonmap[0], 'b' : jsonmap[1], 'c' : jsonmap[2], 'd' : jsonmap[3]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

