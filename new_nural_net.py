import numpy as np
import matplotlib.pyplot as plt
import h5py
from testCases_v2 import *
# import dataset
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_gradient(z):
    return 1 - np.power(tanh(z),2)

def relu(z):
    return np.maximum(0, z)

def relu_gradient(z):
    return np.maximum(0, 1)

# process dataset for models
# vectorize data

def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
   
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # print(np.size(train_set_x_orig[1]))
    # plt.imshow(train_set_x_orig[5])
    # plt.show()
    # ## START CODE HERE ## (PUT YOUR IMAGE NAME)
    # my_image = "my_image.jpg"   # change this to the name of your image file
    # ## END CODE HERE ##

    # # We preprocess the image to fit your algorithm.
    # num_px = 0
    # fname = "images/" + my_image
    # image = np.array(ndimage.imread(fname, flatten=False))
    # my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    # my_predicted_image = predict(d["w"], d["b"], my_image)

    # plt.imshow(image)
    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    m_train = train_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    m_test = test_set_x_orig.shape[0]
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x_orig shape: " + str(train_set_x_orig.shape))
    print ("train_y shape: " + str(train_set_y_orig.shape))
    print ("test_set_x_orig shape: " + str(test_set_x_orig.shape))
    print ("test_y shape: " + str(test_set_y_orig.shape))
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    return train_x, train_set_y_orig, test_x, test_set_y_orig


def initialize_caches(layers_dims):
    caches = {}
    L = len(layers_dims)

    for l in range(1, L):
        caches['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / 100
        caches['b' + str(l)] = np.zeros(shape = (layers_dims[l], 1))
        caches['Z' + str(l)] = 0
        caches['A' + str(l)] = 0
        # caches['g' + str(l)] = 0
    
    # print(caches)
    return caches

def linear_forward(A_prev, W, b):

    return np.dot(W, A_prev) + b

def activation_forward(Z, activation):
    
    activation_func = activation_functions[activation][0]
    return activation_func(Z)

def forward_propagation(X, caches, activation_funcs_map):

    L = len(caches) // 4
    caches["A" + str(0)] = X
    for l in range(1, L + 1):
        A_prev = caches["A" + str(l - 1)]
        W = caches['W' + str(l)]
        b = caches["b" + str(l)]
        Z = linear_forward(A_prev,W, b)
        activation = activation_funcs_map['g' + str(l)]
        A_curr = activation_forward(Z, activation)
        caches['Z' + str(l)] = Z
        caches['A' + str(l)] = A_curr

    return A_curr, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = - 1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    return cost

def single_layer_back_prop(A_prev, da, Z, activation):

    m = Z.shape[1]
    print(da, ' da')

    print(A_prev, " A_prev")
    activation_gradient = activation_functions[activation][1]
    dadz = activation_gradient(Z)
    dz = np.multiply(dadz, da)
    print(dz, " dz")
    dw = np.dot(dz, A_prev.T) / m
    print(dw, ' dw')
    db = np.sum(dz, keepdims = True) / m
    return dz, dw, db

def back_propagation(AL, Y, caches, activation_funcs_map, alpha):

    L = len(caches) // 4
    daL = np.divide(-Y, AL) + np.divide(1 - Y, 1 - AL)
    da = daL
    print(Y, ' Y')

    for l in range(L, L - 1, -1):
        Z = caches["Z" + str(l)]
        W = caches["W" + str(l)]
        b = caches["b" + str(l)]
        A_prev = caches["A" + str(l - 1)]
        activation = activation_funcs_map['g' + str(l)]
        dz, dw, db = single_layer_back_prop(A_prev, da, Z, activation)
        # print(dw[1])
        # print('dw shape')
        da = np.dot(W.T, dz)

        caches["W" + str(l)] = W - np.multiply(alpha, dw)
        caches["b" + str(l)] = b - np.multiply(alpha, db)

    return caches

def train(train_set_x, train_set_y):

    num_iterations = 3000
    n_x = train_set_x.shape[0] * train_set_x.shape[1] * train_set_x[2]
    layers_dims = [12288, 20, 7, 5, 1]
    caches = initialize_caches(layers_dims)
    caches['A0'] = train_set_x
    activation_funcs_map = {'g1':'relu',
                        'g2':'relu',
                        'g3':'relu',
                        'g4':'sigmoid'}
    for i in range(1, num_iterations):
        AL, caches = forward_propagation(train_set_x, caches, activation_funcs_map)
        for l in range(1, len(caches) // 4 + 1):
            # print(caches['W' + str(l)])
            print('b')
            print(caches['b' + str(l)].shape)
        print("w at %i iteration" %i)
        cost = compute_cost(AL, train_set_y)
        print(cost)
        print("cost at %i iteration" % i)
        caches = back_propagation(AL, train_set_y, caches, activation_funcs_map, 0.0075)

def predict(X, Y, caches, activation_funcs_map):

    activation_functions = {"sigmoid":[sigmoid, sigmoid_gradient],
                            "tanh":[tanh, tanh_gradient],
                            "relu":[relu, relu_gradient]}

    activation_funcs_map = {'g1':'relu',
                            'g2':'relu',
                            'g3':'relu',
                            'g4':'sigmoid'}
    AL, caches = forward_propagation(X, caches, activation_funcs_map)
    print(AL)

def image_preprocess():
    my_image = "test1.jpg"
    from skimage.transform import resize
    # We preprocess the image to fit your algorithm.
    fname = my_image
    image = np.array(plt.imread(my_image))
    print(np.size(image))
    plt.imshow(image)
    
    my_image = resize(image, (num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_image_show = resize(image, (num_px, num_px))
    resize
    # my_predicted_image = predict(my_image)
    print(np.size(my_image_show))
    plt.imshow(my_image_show)
    plt.show()
    # print("y = " + str(np.squeeze(my_predicted_image)) + 
    #     ", your algorithm predicts a \"" + 
    #     classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  
    #     "\" picture.")

n_x = 12288     # num_px * num_px * 3
n_h_1 = 5
n_h_2 = 4
# n_h_3 = 5
n_y = 3
layers_dims = [5, 4, 3]

activation_functions = {"sigmoid":[sigmoid, sigmoid_gradient],
                        "tanh":[tanh, tanh_gradient],
                        "relu":[relu, relu_gradient]}

train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()
caches = train(train_set_x, train_set_y)
np.random.seed(1)
num_px = np.power(2,6)
# image_preprocess()
A, W, b = linear_forward_test_case()
# print("AWB")
# print(A, W, b)
# initialize_caches(layers_dims)
z = linear_forward(A, W, b)
# print(z)

# X, parameters = L_model_forward_test_case()
# parameters = {'W1': [[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
#        [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
#        [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
#        [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]], 
#        'W3': [[ 0.9398248 ,  0.42628539, -0.75815703]], 
#         'b3': [[-0.16236698]], 'b2': [[ 1.50278553],
#        [-0.59545972],
#        [ 0.52834106]], 'b1': [[ 1.38503523],
#        [-0.51962709],
#        [-0.78015214],
#        [ 0.95560959]], 'W2': [[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
#        [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
#        [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]}
# X = [[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
#  [-2.48678065,  0.91325152 , 1.12706373, -1.51409323],
#  [ 1.63929108, -0.4298936 ,  2.63128056,  0.60182225],
#  [-0.33588161,  1.23773784 , 0.11112817,  0.12915125],
#  [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]]

# AL = forward_propagation(X, parameters, activation_funcs_map)
# AL, Y_assess, caches = L_model_backward_test_case()
# print(AL, Y_assess, caches)
