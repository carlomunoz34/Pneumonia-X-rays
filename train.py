from CNN import ConvolutionalNetwork, ConvolutionalNetwork2
from utils import get_test2, get_next_batch, get_val, get_test_26, get_test_78
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

    model = ConvolutionalNetwork2()
    print("Starting random search at:", datetime.now())

    t0 = datetime.now()
    for i in range(iterations):
        tf.random.set_random_seed(SEED)

        #Choose hyperparams
        dict = get_random_dict(learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

        print("%i of %i:" %(i+1, iterations), dict)
        
        model.ensamble((200, 200, 3), conv_shapes=dict["conv_shape"], 
                    vanilla_shapes=dict["vanilla_shape"], activation=dict["activation"])

        score = model.fit(get_next_batch, learning_rate=dict["learing rate"], beta1=0.9, beta2=0.999, 
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


def grid_search(learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    best_score = float('inf')
    best_dict = ""
    i = 1
    total = len(learning_rates) * len(epochs) * len(conv_shapes) * len(vanilla_shapes) * len(activations)

    t0 = datetime.now()
    model = ConvolutionalNetwork()
    
    for learning_rate in learning_rates:
        for epoch in epochs:
            for conv_shape in conv_shapes:
                for vanilla_shape in vanilla_shapes:
                    for activation in activations:
                        tf.random.set_random_seed(SEED)

                        dict = {"learing rate": learning_rate,
                                "epoch": epoch, 
                                "conv_shape": conv_shape,
                                "vanilla_shape": vanilla_shape,
                                "activation": activation}

                        msg = "%i of %i: " %(i, total) + str(dict)
                        
                        print(msg)
                        i += 1

                        if i <= 6:
                            continue

                        model.ensamble((128, 128, 3), 2, conv_shapes=conv_shape, 
                                vanilla_shapes=vanilla_shape, activation=activation)

                        score = model.fit(get_next_batch, learning_rate=learning_rate, batch_number=134,
                                epochs=epoch, verbose=False, next_test_batch=get_test_78, test_batch_number=78,
                                best=best_score)
                        
                        dict['cost'] = score
                        dict['seed'] = SEED

                        if score < best_score:
                            best_score = score
                            best_dict = dict

                            best_model_json = json.dumps(best_dict)
                            f = open("../Models/best_model.json", "w")
                            f.write(best_model_json)
                            f.close()
                        
                        print("Cost:", score, "Best cost:", best_score)
                        gc.collect()
    
    t1 = datetime.now()
    print("Random search finished. Best model saved in /Models/best/")
    print("Elapsed time: ", t1 - t0)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    learning_rates = [0.001, 0.0001]
    epochs = [5, 8, 10]
    conv_shapes = [(32, 64), (64, 128), (64, 64), (128, 128), (30, 60, 90)]
    vanilla_shapes = [(1024,), (512, 512), (512, 256, 128)]
    activations = ['relu', 'tanh']

    #learning_rates = [0.001]
    #epochs = [10]
    #conv_shapes = [(30, 60, 90)]
    #vanilla_shapes = [(512, 256)]
    #activations = ['relu']


    #grid_search(learning_rates, epochs, conv_shapes, vanilla_shapes, activations)
    random_search(4, learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

    