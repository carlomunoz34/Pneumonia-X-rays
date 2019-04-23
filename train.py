from CNN import ConvolutionalNetwork
from utils import get_test2, get_next_batch, get_val, get_test_26, get_test_78
from matplotlib import pyplot as plt
import numpy as np
import json

def random_search(iterations, learning_rates, epochs, conv_shapes, vanilla_shapes, activations):
    best_score = float('inf')
    best_dict = None

    for _ in range(iterations):
        #Choose hyperparams
        dict = {"learing rate": learning_rates[np.random.randint(0, len(learning_rates))],
                "epoch": epochs[np.random.randint(0, len(epochs))], 
                "conv_shape": conv_shapes[np.random.randint(0, len(conv_shapes))],
                "vanilla_shape": vanilla_shapes[np.random.randint(0, len(vanilla_shapes))],
                "activation": activations[np.random.randint(0, len(activations))]}
        
        model = ConvolutionalNetwork((128, 128, 3), 2, conv_shapes=dict["conv_shape"], 
                    vanilla_shapes=dict["vanilla_shape"], activation=dict["activation"])

        history = model.fit(get_next_batch, learning_rate=dict["learing rate"], beta1=0.9, beta2=0.999, 
                    batch_number=2, epochs=dict["epoch"], verbose=False, next_test_batch=get_test_78, 
                    test_batch_number=78)

        if history[-1] < best_score:
            best_score = history[-1]
            best_dict = dict
            model.save_model("../Models/best_model.ckpt")
            
    best_model_json = json.dumps(best_dict)
    f = open("best_model.json", "w")
    f.write(best_model_json)
    f.close()

if __name__ == "__main__":
    learning_rates = [0.001, 0.0001]
    epochs = [3, 4]
    conv_shapes = [(32, 64), (64, 128)]
    vanilla_shapes = [(1024,), (512, 512), (128, 256, 512)]
    activations = ['relu', 'relu', 'tanh', 'relu']

    random_search(1, learning_rates, epochs, conv_shapes, vanilla_shapes, activations)