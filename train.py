from CNN import ConvolutionalNetwork
from utils import get_test2, get_next_batch, get_val, get_test_26, get_test_78
from matplotlib import pyplot as plt
import numpy as np
import json
from datetime import datetime
import gc
import sys
import tensorflow as tf

def random_search(iterations, learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    best_score = float('inf')
    best_dict = None

    model = ConvolutionalNetwork()

    for _ in range(iterations):
        #Choose hyperparams
        dict = {"learing rate": learning_rates[np.random.randint(0, len(learning_rates))],
                "epoch": epochs[np.random.randint(0, len(epochs))], 
                "conv_shape": conv_shapes[np.random.randint(0, len(conv_shapes))],
                "vanilla_shape": vanilla_shapes[np.random.randint(0, len(vanilla_shapes))],
                "activation": activations[np.random.randint(0, len(activations))],
                "cost": 0.0}
        
        model.ensamble((128, 128, 3), 2, conv_shapes=dict["conv_shape"], 
                    vanilla_shapes=dict["vanilla_shape"], activation=dict["activation"])

        history = model.fit(get_next_batch, learning_rate=dict["learing rate"], beta1=0.9, beta2=0.999, 
                    batch_number=2, epochs=dict["epoch"], verbose=False, next_test_batch=get_test_78, 
                    test_batch_number=78)

        dict['cost'] = history[-1]

        if history[-1] < best_score:
            best_score = history[-1]
            best_dict = dict
            model.save_model("../Models/best_model.ckpt")

            best_model_json = json.dumps(best_dict)
            f = open("best_model.json", "w")
            f.write(best_model_json)
            f.close()    


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

                        history = model.fit(get_next_batch, learning_rate=learning_rate, batch_number=134,
                                epochs=epoch, verbose=False, next_test_batch=get_test_78, test_batch_number=78,
                                best=best_score)
                        
                        dict['cost'] = history[-1]

                        if history[-1] < best_score:
                            best_score = history[-1]
                            best_dict = dict

                            best_model_json = json.dumps(best_dict)
                            f = open("../Models/best_model.json", "w")
                            f.write(best_model_json)
                            f.close()
                        
                        gc.collect()
    
    t1 = datetime.now()
    print("Random search finished. Best model saved in /Models/best/")
    print("Elapsed time: ", t1 - t0)


if __name__ == "__main__":
    tf.random.set_random_seed(34)
    np.random.seed(35)

    #learning_rates = [0.001]
    #epochs = [3]
    #conv_shapes = [(32, 64), (64, 128)]
    #vanilla_shapes = [(1024,), (512, 512), (512, 256, 128)]
    #activations = ['relu']

    #grid_search(learning_rates, epochs, conv_shapes, vanilla_shapes, activations)

    model = ConvolutionalNetwork()
    model.ensamble((128, 128, 3), 2, conv_shapes=(64, 128), 
            vanilla_shapes=(512, 512))

    history = model.fit(get_next_batch, learning_rate=0.001, batch_number=134,
            epochs=3, verbose=False, next_test_batch=get_test_78, test_batch_number=78,
            best=float('inf'))
    
    print("Cost:", history[-1])

    