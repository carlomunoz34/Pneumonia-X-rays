import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical

def get_data(img_size, train_limit= None, test_limit= None):
    print("Getting the data")

    #Get Train set
    Xtrain = []
    Ytrain = []

    train_normal_files = glob("../chest_xray/train/NORMAL/*.jpeg")
    print("Getting train data")
    print("Starting with normal images")

    i = 0
    for image in train_normal_files:
        if train_limit is not None and i >= train_limit // 2: 
            break
        i += 1

        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        Xtrain.append(img.astype(np.float32))
        Ytrain.append(0)

    train_phneumonia_files = glob("../chest_xray/train/PNEUMONIA/*.jpeg")
    print("Starting with phneumonia images")

    i = 0
    for image in train_phneumonia_files:
        if train_limit is not None and i >= train_limit // 2: 
            break
        i += 1

        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtrain.append(img.astype(np.float32))
        Ytrain.append(1)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Ytrain = to_categorical(Ytrain, num_classes=2)
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    #Get test data
    Xtest = []
    Ytest = []

    print("Getting test data")
    print("Starting with normal images")

    test_normal_files = glob("../chest_xray/test/NORMAL/*.jpeg")

    i = 0
    for image in test_normal_files:
        if test_limit is not None and i >= test_limit // 2: 
            break
        i += 1

        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(0)

    train_phneumonia_files = glob("../chest_xray/test/PNEUMONIA/*.jpeg")
    print("Starting with phneumonia images")

    i = 0
    for image in train_phneumonia_files:
        if test_limit is not None and i >= test_limit // 2: 
            break
        i += 1

        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(1)

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    Ytest = to_categorical(Ytest, num_classes=2)
    Xtest, Ytest = shuffle(Xtest, Ytest)

    print("All data read")

    return Xtrain, Xtest, Ytrain, Ytest


def get_test(img_size=128):
    #Get test data
    Xtest = []
    Ytest = []

    #print("Getting test data")
    #print("Starting with normal images")

    test_normal_files = glob("../chest_xray/test/NORMAL/*.jpeg")

    for image in test_normal_files:
        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(0)

    train_phneumonia_files = glob("../chest_xray/test/PNEUMONIA/*.jpeg")
    #print("Starting with phneumonia images")

    for image in train_phneumonia_files:
        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(1)

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    #Ytest = to_categorical(Ytest, num_classes=2)
    #Xtest, Ytest = shuffle(Xtest, Ytest)

    #print("All data read")

    return Xtest, Ytest


def one_hot(y):
    K = len(set(y))
    N = len(y)
    ind = np.zeros((N, K))
    for i in y:
        ind[i][y[i]] = 1
    return ind


def get_next_batch(epoch, img_size=128):
    Xtrain = []
    Ytrain = []

    #Process normal images
    train_normal_files = glob("../chest_xray/train/NORMAL/*.jpeg")
    normal_lower = epoch * 10
    normal_upper = normal_lower + 10

    for i in range(normal_lower, normal_upper):
        img = cv2.imread(train_normal_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        Xtrain.append(img.astype(np.float32))
        Ytrain.append(0)


    #Process pneumonia images
    train_phneumonia_files = glob("../chest_xray/train/PNEUMONIA/*.jpeg")
    pneumonia_lower = epoch * 28
    pneumonia_upper = pneumonia_lower + 28

    for i in range(pneumonia_lower, pneumonia_upper):
        img = cv2.imread(train_phneumonia_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtrain.append(img.astype(np.float32))
        Ytrain.append(1)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    #Ytrain = to_categorical(Ytrain, num_classes=2)
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain = np.reshape(Ytrain, list(Ytrain.shape) + [1])

    return Xtrain, Ytrain


def get_val(img_size=128):
    #Get test data
    Xtest = []
    Ytest = []

    print("Getting test data")
    print("Starting with normal images")

    test_normal_files = glob("../chest_xray/val/NORMAL/*.jpeg")

    for image in test_normal_files:
        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(0)

    train_phneumonia_files = glob("../chest_xray/val/PNEUMONIA/*.jpeg")
    print("Starting with phneumonia images")

    for image in train_phneumonia_files:
        img = cv2.imread(image)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtest.append(img.astype(np.float32))
        Ytest.append(1)

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    #Ytest = to_categorical(Ytest, num_classes=2)
    Xtest, Ytest = shuffle(Xtest, Ytest)

    print("All data read")

    return Xtest, Ytest


def get_test2(epoch, img_size=128):
    Xtrain = []
    Ytrain = []

    #Process normal images
    train_normal_files = glob("../chest_xray/test/NORMAL/*.jpeg")
    normal_lower = epoch * 39
    normal_upper = normal_lower + 39 if normal_lower + 39 < len(train_normal_files) else len(train_normal_files) - 1

    for i in range(normal_lower, normal_upper):
        img = cv2.imread(train_normal_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        Xtrain.append(img.astype(np.float32))
        Ytrain.append(0)


    #Process pneumonia images
    train_phneumonia_files = glob("../chest_xray/test/PNEUMONIA/*.jpeg")
    pneumonia_lower = epoch * 65
    pneumonia_upper = pneumonia_lower + 65 if pneumonia_lower + 65 < len(train_phneumonia_files) else len(train_phneumonia_files) -1

    for i in range(pneumonia_lower, pneumonia_upper):
        img = cv2.imread(train_phneumonia_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtrain.append(img.astype(np.float32))
        Ytrain.append(1)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    #Ytrain = to_categorical(Ytrain, num_classes=2)
    #Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Ytrain = np.reshape(Ytrain, list(Ytrain.shape) + [1])

    return Xtrain, Ytrain


def get_test_26(epoch, img_size=128):
    Xtrain = []
    Ytrain = []

    #Process normal images
    train_normal_files = glob("../chest_xray/test/NORMAL/*.jpeg")
    normal_lower = epoch * 9
    normal_upper = normal_lower + 9 if normal_lower + 9 < len(train_normal_files) else len(train_normal_files) - 1

    for i in range(normal_lower, normal_upper):
        img = cv2.imread(train_normal_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        Xtrain.append(img.astype(np.float32))
        Ytrain.append(0)


    #Process pneumonia images
    train_phneumonia_files = glob("../chest_xray/test/PNEUMONIA/*.jpeg")
    pneumonia_lower = epoch * 15
    pneumonia_upper = pneumonia_lower + 15 if pneumonia_lower + 15 < len(train_phneumonia_files) else len(train_phneumonia_files) -1

    for i in range(pneumonia_lower, pneumonia_upper):
        img = cv2.imread(train_phneumonia_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtrain.append(img.astype(np.float32))
        Ytrain.append(1)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    #Ytrain = to_categorical(Ytrain, num_classes=2)
    #Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    return Xtrain, Ytrain


def get_test_78(epoch, img_size=128):
    Xtrain = []
    Ytrain = []

    #Process normal images
    train_normal_files = glob("../chest_xray/test/NORMAL/*.jpeg")
    normal_lower = epoch * 3
    normal_upper = normal_lower + 3 if normal_lower + 3 < len(train_normal_files) else len(train_normal_files) - 1

    for i in range(normal_lower, normal_upper):
        img = cv2.imread(train_normal_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        Xtrain.append(img.astype(np.float32))
        Ytrain.append(0)


    #Process pneumonia images
    train_phneumonia_files = glob("../chest_xray/test/PNEUMONIA/*.jpeg")
    pneumonia_lower = epoch * 5
    pneumonia_upper = pneumonia_lower + 5 if pneumonia_lower + 5 < len(train_phneumonia_files) else len(train_phneumonia_files) -1

    for i in range(pneumonia_lower, pneumonia_upper):
        img = cv2.imread(train_phneumonia_files[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        Xtrain.append(img.astype(np.float32))
        Ytrain.append(1)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    #Ytrain = to_categorical(Ytrain, num_classes=2)
    #Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain = np.reshape(Ytrain, list(Ytrain.shape) + [1])

    return Xtrain, Ytrain