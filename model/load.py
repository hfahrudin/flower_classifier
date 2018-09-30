# -*- coding: utf-8 -*-

from keras.models import model_from_json
import tensorflow as tf


def init() :
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam')
    
    graph = tf.get_default_graph()
    
    return loaded_model, graph

