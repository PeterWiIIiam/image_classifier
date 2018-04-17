import numpy as np
import matplotlib.pyplot as plt
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
    return max(0, z)

def relu_gradient():
    return max(0, 1)

activation_functions = {"sigmoid":[sigmoid, sigmoid_gradient],
                        "tanh":[tanh, tanh_gradient],
                        "relu":[relu, relu_gradient]}

# process dataset for models
# vectorize data

def load_dataset():
#    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
#    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
#    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
#    
#    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
#    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
#    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
#    
#    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#    
#    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


    ## START CODE HERE ## (PUT YOUR IMAGE NAME)
    my_image = "my_image.jpg"   # change this to the name of your image file
    ## END CODE HERE ##

    # We preprocess the image to fit your algorithm.
    num_px = 0
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros(shape = (l, 1))
        parameters['Z' + str(l)] = 0
    
    return parameters

def linear_forward(W, b, A_prev):

    return np.dot(W, A_prev) + b

def activation_forward(W, b, A_prev, activation):
    
    activation_func = activation_functions[activation][0]
    return activation_func(linear_forward(W, b, A_prev))

def forward_propagation(parameters, X, activation_funcs_map):

    L = len(parameters) // 3
    
    A_prev = X
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parametersh["W" + str(l)]
        activation = activation_funcs_map[str(l)]
        A_curr = activation_forward(W, b, A_prev, activation)
        A_prev = A_curr

    return A_curr

def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = - 1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

def single_layer_back_prop(A_prev, da_next, Z, activation):
    
    activation_gradient = activation_functions[activation][1]
    dadz = activation_gradient(Z)
    dz = np.multiply(dadz, dz)
    dw = np.dot(dz, A_prev.T)
    db = np.sum(dz, axis = 1)
    return dw, db

def back_propagation(parameters, activation_funcs_map):

    L = len(parameters) // 3
    daL =


    for l in reversed(range(L - 1)):
        Z =
