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

def relu_gradient():
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

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def initialize_caches(layers_dims):
    caches = {}
    L = len(layers_dims)
    np.random.seed(3)

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

def activation_forward(A_prev, W, b, activation):
    
    activation_func = activation_functions[activation][0]
    return activation_func(linear_forward(A_prev,W, b))

def forward_propagation(X, caches, activation_funcs_map):
    print(caches)

    L = len(caches) // 4
    caches["A" + str(0)] = X
    A_prev = X
    for l in range(1, 3 ):
        W = caches['W' + str(l)]
        b = caches["b" + str(l)]
        activation = activation_funcs_map['g' + str(l)]
        A_curr = activation_forward(A_prev, W, b, activation)
        caches['A' + str(l)] = A_curr
        A_prev = A_curr

    return A_curr

def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = - 1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    return cost

def single_layer_back_prop(A_prev, da, Z, activation):

    activation_gradient = activation_functions[activation][1]
    dadz = activation_gradient(Z)
    dz = np.multiply(dadz, dz)
    dw = np.dot(dz, A_prev.T)
    db = np.sum(dz, axis = 1)
    return dz, dw, db

def back_propagation(AL, caches, activation_funcs_map, alpha):

    L = len(caches) // 4
    daL = np.divide(-Y, AL) + np.divide(1 - Y, 1 - AL)
    da = daL

    for l in range(L, 0, -1):
        Z = caches["Z" + str(l)]
        W = caches["W" + str(l)]
        b = caches["b" + str(l)]
        A_prev = caches["A" + str(l - 1)]
        activation = activation_funcs_map[l]
        dz, dw, db = single_layer_back_prop(A_prev, da, Z, activation)
        da = np.dot(W.T, dz)

        caches["W" + str(l)] = W - np.multiply(alpha, W)
        caches["b" + str(l)] = b - np.multiply(alpha, b)

def predict(image):

	activation_functions = {"sigmoid":[sigmoid, sigmoid_gradient],
	                        "tanh":[tanh, tanh_gradient],
	                        "relu":[relu, relu_gradient]}

	activation_funcs_map = {'g1':'relu',
							'g2':'sigmoid',
							'g3':'relu',
							'g4':'sigmoid'}
	n_x = 12288     # num_px * num_px * 3
	n_h_1 = 5
	n_h_2 = 4
	# n_h_3 = 5
	n_y = 3
	layers_dims = [n_x, n_h_1, n_h_2, n_y]


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

train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()
np.random.seed(1)
num_px = np.power(2,6)
# image_preprocess()
A, W, b = linear_forward_test_case()
# print("AWB")
# print(A, W, b)
# initialize_caches(layers_dims)
z = linear_forward(A, W, b)
# print(z)

activation_functions = {"sigmoid":[sigmoid, sigmoid_gradient],
                        "tanh":[tanh, tanh_gradient],
                        "relu":[relu, relu_gradient]}

activation_funcs_map = {'g1':'relu',
                        'g2':'sigmoid',
                        'g3':'relu',
                        'g4':'sigmoid'}
X, parameters = L_model_forward_test_case()
AL = forward_propagation(X, parameters, activation_funcs_map)
print(AL)


