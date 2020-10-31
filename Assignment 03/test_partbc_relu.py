import numpy as np
from mnist import MNIST
import pickle



def relu_activation(data_array):
    return np.maximum(data_array, 0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def feedforward(x,biases,weights):
    """Return the output of the network if "a" is input."""
    #print "x  initial",np.shape(a)
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        #print "z ",z
        zs.append(z)
        activation = relu_activation(z)
        #print "length ",len(activation)
        activations.append(activation)
    activations[-1]=softmax(zs[-1])   
    return activations[-1]


def evaluate(test_data,test_labels,biases,weights):
    test_results = [(np.argmax(feedforward(x,biases,weights))) for x in test_data]
    return sum([ int(i==j) for i,j in zip(test_results,test_labels)])


def predict(dataset,labels,biases,weights):
    score=float(evaluate(dataset,labels,biases,weights))/len(dataset)
    print "score ", score



mndata = MNIST('/home/garvita/Desktop/HW-3-NN')
te_d,te_l=mndata.load_testing()
ted=[np.reshape(x, (784, 1)) for x in te_d]

dataset=[]
labels=[]
for i in range(len(te_d)):
    dataset.append(te_d[i])
    labels.append(te_l[i])
    

dataset=np.array(dataset)
labels=np.array(labels)
biases=[]
weigths=[]
with open("partbc_relu_biases.pkl","r") as f:
    biases=pickle.load(f)
    #print type(var)
with open("partbc_relu_weights.pkl","r") as f:
    weights=pickle.load(f)
predict(ted,labels,biases,weights)
