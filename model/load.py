# -*- coding: utf-8 -*-

from keras.models import model_from_json
import tensorflow as tf

def init() :
    json_path = ['json/model_resnet50.json', 'json/model_vgg16.json', 'json/model_xception.json', 'json/model_inception.json']
    weight_path = ['weight/model_resnet50.h5', 'weight/model_vgg16.h5', 'weight/model_xception.h5', 'weight/model_inception.h5']
    all_model = []
    for i in range(0, len(json_path)):
        json_file = open(json_path[i], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_path[i])
        print("Loaded model from disk")
        loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam')
        all_model.append(loaded_model)
    
    graph = tf.get_default_graph()
    
    return all_model, graph
