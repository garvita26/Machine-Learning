import h5py
import numpy as np
from mnist import MNIST
from sklearn.model_selection import KFold
import pickle
import itertools

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu_activation(data_array):
    return np.maximum(data_array, 0)


def relu_prime(z):
    return 1.0 / (1.0 + np.exp(-z))

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


def SGD(training_data,training_labels, epochs, mini_batch_size, eta,biases,weights,num_layers,test_data,test_labels):
    if test_data: n_test = len(test_data)
    #print training_labels
    n = len(training_data)
    accuracy_K=[]
    for j in xrange(epochs):
        
        mini_batches_data = [training_data[k:k+mini_batch_size]for k in xrange(0, n, mini_batch_size)]
        #print "mini batch data ",len(mini_batches_data)
        mini_batches_labels = [training_labels[k:k+mini_batch_size]for k in xrange(0, n, mini_batch_size)]
        for i in range(len(mini_batches_data)):
            #print "size ",np.shape(mini_batches_labels[i])
            biases,weights=update_mini_batch(mini_batches_data[i],mini_batches_labels[i], eta,biases,weights,num_layers)
        if test_data:
            score=float(evaluate(test_data,test_labels,biases,weights))/n_test
            print "Epoch {0}:{1}".format(
                    j, score)
            temp=[]
            temp.append(j)
            temp.append(score)
            accuracy_K.append(temp)
        else:
            print "Epoch {0} complete".format(j)    
        
    return biases,weights,accuracy_K    



def backprop( x, y,biases,weights,num_layers):
    #print "y ",y
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
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
    # backward pass
    
    delta = cost_derivative(activations[-1], y)
    
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #print "delta w ",np.dot(delta, activations[-2].transpose())
    for l in xrange(2, num_layers):
        z = zs[-l]
        sp = relu_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)


def update_mini_batch(mini_batch,mini_label, eta,biases,weights,num_layers):
    nabla_b = [np.zeros(b.shape) for b in biases]
    #print "nabla_b ",np.shape(nabla_b[0])
    nabla_w = [np.zeros(w.shape) for w in weights]
    
    #print "mb ",len(mini_batch[0])
    for i in range(len(mini_batch)):
        #print "ml ",mini_label[i]
        delta_nabla_b, delta_nabla_w = backprop(mini_batch[i],mini_label[i],biases,weights,num_layers )
        #print "delta b ",delta_nabla_b
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
        biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b)]
    return biases,weights    


def evaluate(test_data,test_labels,biases,weights):
    test_results = [(np.argmax(feedforward(x,biases,weights))) for x in test_data]
    return sum([ int(i==j) for i,j in zip(test_results,test_labels)])

def cost_derivative(output_activations, y):
    #print "op ",(output_activations)
    #print "y ",(y)
    #print ((output_activations)-y)
    return ((output_activations)-y)

def grid_search(grid):
    g=[]
    m=[]
    for i in grid.keys():
        
        g.append(grid[i])
    m=list(itertools.product(*g))
    return m   

def vectorized_result(y):
    e = np.zeros((10,1))
    
    e[y]=1.0
        
    return e

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

mndata = MNIST('/home/garvita/Desktop/HW-3-NN')
#f = gzip.open('../data/mnist.pkl.gz', 'rb')
tr_d,tr_l=mndata.load_training()
te_d,te_l=mndata.load_testing()
dataset=[]
labels=[]
for i in range(len(tr_d)):
    dataset.append(tr_d[i])
    labels.append(tr_l[i])
    

dataset=np.array(dataset)
labels=np.array(labels)

test_inputs = [np.reshape(x, (784, 1)) for x in te_d]
#test_data = zip(test_inputs, te_l)
sizes=[784,100,50,10]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
#print "bias " ,np.shape(biases[2])
weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
grid = {'batch_size': [10,20,50],'max_iter': [10,50,100],'eta' : [.001,.00001,.000001]}
m=grid_search(grid)
kf = KFold(n_splits=5)
accuracy_allfold=[]
biases_all=[]
weights_all=[]
for i in list(m):
    for train_index, test_index in kf.split(dataset):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = dataset[train_index], dataset[test_index]
        td=[np.reshape(x, (784, 1)) for x in X_train]
        ted=[np.reshape(x, (784, 1)) for x in X_test]
        y_train, y_test = labels[train_index], labels[test_index]
        tl=[vectorized_result(y) for y in y_train]
        tel=labels[test_index]
        biases,weights,accuracy_K=SGD(td, tl,i[1], i[0], i[2],biases,weights,num_layers,ted,tel)
        accuracy_allfold.append(accuracy_K)
        biases_all.append(biases)
        weights_all.append(weights)

index=0
for i in range(len(accuracy_allfold)):
    max_val=0
    index=0
    if accuracy_allfold[i][-1][1]>max_val:
        max_val=accuracy_allfold[i][-1][1]
        index=i

#print "index ",index
with open("partbc_relu__weights.pkl","wb") as f:
    pickle.dump(weights_all[index],f)

with open("partbc_relu_biases.pkl","wb") as f:
    pickle.dump(biases_all[index],f)
with open("epoch_accuracy_partbc_relu.txt","w") as f:
    for i in accuracy_allfold:
        f.write(str(i))
        f.write("\n")



#biases,weights=SGD(training_inputs, training_labels,50, len(training_inputs), .000005,biases,weights,num_layers,test_inputs,te_l)
