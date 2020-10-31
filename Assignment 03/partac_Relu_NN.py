import h5py
import numpy as np
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
    

def feedforward(test_data,biases,weights):
    """Return the output of the network if "a" is input."""
    #print "x  initial",np.shape(a)
    r=[]
    for i in test_data:
        x=i
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
        activations[-1]=sigmoid(zs[-1]) 
        r.append(np.argmax(activations[-1]))
    return r


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
        if test_data and j%10==0:
            score= evaluate(test_data,test_labels,biases,weights)
            print "Epoch {0}: {1}".format(
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
        #print "w shape ",np.shape(w)
        #print "act shape ",np.shape(activation)
        #print "b shape ",np.shape(b)
        #print "dot " ,np.shape(np.dot(w, activation))
        z = np.dot(w, activation)+b
        #print "z ",z
        zs.append(z)
        activation = relu_activation(z)
        #print "length ",len(activation)
        activations.append(activation)
    #print "activation ",np.shape(activations[3])    
    # backward pass
    #print "a[-1] ",np.shape(activations[-1])
    activations[-1]=sigmoid(zs[-1])
    delta = cost_derivative(activations[-1], y)
    #print "delta ",delta
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

    test_results=feedforward(test_data,biases,weights)
        
    
    result=[]
    #print test_results
    count7=0
    count9=0
    for i in test_results:
        #print i
        if i==0:
            count7+=1
            result.append(7)
        else:
            count9+=1
            result.append(9)
    
    count=0
    for i in range(len(test_labels)):
        
        if(result[i]==test_labels[i]):
            #print "hi"
            count=count+1
                    
    #print count                
    return (float(count)/len(test_labels))


def grid_search(grid):
    g=[]
    m=[]
    for i in grid.keys():
        
        g.append(grid[i])
    m=list(itertools.product(*g))
    return m   

def cost_derivative(output_activations, y):
    #print "op ",(output_activations)
    #print "y ",(y)
    #print ((output_activations)-y)
    return ((output_activations)-y)



def vectorized_result(y):
    e = np.zeros((2,1))
    
    if y==7:
        
        e[0]=1.0
        e[1]=0.0
    else:
        e[0]=0.0
        e[1]=1.0
        
    return e

f=h5py.File("dataset_partA.h5","r+")
#print f
#print "keys: %s" %f.keys()
dataset =(f['X'][:])
#print type(dataset)
dt=[]
for i in range(len(dataset)):
    temp=(dataset[i,:,:])
    temp=temp.flatten()
    dt.append(temp)

dt=np.array(dt)    
#print "dt ",np.shape(dt) 
labels=f['Y'][:]
l=[]

for i in labels:
    
    l.append(vectorized_result(i))


tl=[vectorized_result(y) for y in labels]
sizes=[784,100,50,2]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
#print "bias " ,np.shape(biases[2])
weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
grid = {'batch_size': [10,20],'max_iter': [50,100,150],'eta' : [.01,.001,.0001]}
m=grid_search(grid)
#print "weights ", np.shape(weights[2])
#print td
kf = KFold(n_splits=5)
accuracy_allfold=[]
biases_all=[]
weights_all=[]
for i in list(m):
    for train_index, test_index in kf.split(dt):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = dt[train_index], dt[test_index]
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
with open("partac_relu_weights.pkl","wb") as f:
    pickle.dump(weights_all[index],f)

with open("partac_relu__biases.pkl","wb") as f:
    pickle.dump(biases_all[index],f)
with open("epoch_accuracy_partac_relu.txt","w") as f:
    for i in accuracy_allfold:
        f.write(str(i))
        f.write("\n")

