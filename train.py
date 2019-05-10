from CNN import ConvolutionalNetwork, ConvolutionalNetwork2
from utils import get_test2, get_next_batch2, get_val, get_test_26, get_test_78
from matplotlib import pyplot as plt
import numpy as np
import json
from datetime import datetime
import gc
import sys
import tensorflow as tf
import os

SEED = 3435

def get_best():
    if not os.path.isfile("best_model.json"):
        return float('inf')

    f = open("best_model.json")
    dict_str = f.read()
    f.close()
    d = json.loads(dict_str)

    return d['cost']


def get_random_dict(learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    dict = {"learing rate": learning_rates[np.random.randint(0, len(learning_rates))],
            "epoch": epochs[np.random.randint(0, len(epochs))], 
            "conv_shape": conv_shapes[np.random.randint(0, len(conv_shapes))],
            "vanilla_shape": vanilla_shapes[np.random.randint(0, len(vanilla_shapes))],
            "activation": activations[np.random.randint(0, len(activations))]}

    return dict


def random_search(iterations, learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    best_score = get_best()
    best_dict = None

    print("Starting random search at:", datetime.now())

    t0 = datetime.now()
    for i in range(iterations):
        tf.random.set_random_seed(SEED)
        model = ConvolutionalNetwork()

        #Choose hyperparams
        dict = get_random_dict(learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

        print("%i of %i:" %(i+1, iterations), dict)
        
        model.ensamble((128, 128, 3), 2, conv_shapes=dict["conv_shape"], 
                    vanilla_shapes=dict["vanilla_shape"], activation=dict["activation"])

        score = model.fit(get_next_batch2, learning_rate=dict["learing rate"], beta1=0.9, beta2=0.999, 
                    batch_number=134, epochs=dict["epoch"], verbose=False, next_test_batch=get_test_78, 
                    test_batch_number=78, best=best_score)

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

    learning_rates = [0.001, 0.0001]
    epochs = [8, 9, 10]
    conv_shapes = [(32, 64), (64, 128), (64, 64), (128, 128), (30, 60, 90)]
    vanilla_shapes = [(1024,), (512, 512), (512, 256, 128)]
    activations = ['relu', 'tanh']

    random_search(5, learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

    