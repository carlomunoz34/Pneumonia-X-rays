from CNN import ConvolutionalNetwork, ConvolutionalNetwork2
from VGG10 import VGG10
from VGG16 import VGG16
from utils import get_next_batch_424, get_test_78, get_next_batch_848
from matplotlib import pyplot as plt
import numpy as np
import json
from datetime import datetime
import gc
import sys
import tensorflow as tf
import os

SEED = 3435
IMG_SIZE = 150

def get_best():
    if not os.path.isfile("best_model.json"):
        return float('inf')

    f = open("best_model.json")
    dict_str = f.read()
    f.close()
    d = json.loads(dict_str)

    return d['cost']


def get_random_dict(learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    dict = {"learing rate": learning_rates[np.random.randint(len(learning_rates))],
            "epoch": epochs[np.random.randint(len(epochs))],
            "conv_shape": conv_shapes[np.random.randint(len(conv_shapes))],
            "vanilla_shape": vanilla_shapes[np.random.randint(len(vanilla_shapes))],
            "activation": activations[np.random.randint(len(activations))]}

    return dict


def random_search(iterations, learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    best_score = get_best()
    best_dict = None

    print("Starting random search at:", datetime.now())
    print("With a best score of:", best_score)

    t0 = datetime.now()
    for i in range(iterations):
        tf.random.set_random_seed(SEED)
        model = VGG16()

        #Choose hyperparams
        dict = get_random_dict(learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

        print("%i of %i:" %(i+1, iterations), dict)
        
        model.assemble((IMG_SIZE, IMG_SIZE, 3), 2, conv_shapes=dict["conv_shape"], 
                    vanilla_shapes=dict["vanilla_shape"], activation=dict["activation"])

        score = model.fit(get_next_batch_848, learning_rate=dict["learing rate"], 
                    batch_number=848, epochs=dict["epoch"], next_test_batch=get_test_78, 
                    test_batch_number=78, best=best_score, show_percentage=True)

        dict['cost'] = score
        dict['seed'] = SEED

        if score < best_score:
            best_score = score
            best_dict = dict
            
            best_model_json = json.dumps(best_dict)
            f = open("best_model.json", "w")
            f.write(best_model_json)
            f.close()
        
        print("Cost:", score, "Best cost:", best_score)
        gc.collect()
    
    print("Random search finished. Elapsed time:", datetime.now() - t0)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config) 


    #learning_rates = [0.001, 0.0001]
    #epochs = [8, 9, 10]
    #conv_shapes = [(32, 64), (64, 128), (64, 64), (128, 128), (30, 60, 90)]
    #vanilla_shapes = [(1024,), (512, 512), (512, 256, 128)]
    #activations = ['relu', 'tanh']

    learning_rates = [1e-4]
    epochs = [5]
    conv_shapes = [(64, 128, 256, 512, 512)]
    vanilla_shapes = [(4096, )]
    activations = ['relu']

    random_search(1, learning_rates, epochs, conv_shapes, vanilla_shapes, activations)
    