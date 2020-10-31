import numpy as np
import h5py
import pickle

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))




def feedforward(test_data,biases,weights):
    """Return the output of the network if "a" is input."""
    #print "x  initial",np.shape(a)
    r=[]
    for i in test_data:
        a=i
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
            #print "x ",np.shape(np.dot(w, a)+b)
            #print "a ",np.shape(a)
            #print "a ",a
        #print "argmax in ff ",np.argmax(a)
        r.append(np.argmax(a))
    return r


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


def predict(dataset,labels,biases,weights):
    score=evaluate(dataset,labels,biases,weights)
    print "score ", score



f=h5py.File("/home/garvita/Desktop/HW-3-NN/dataset_partA.h5","r+")
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

ted=[np.reshape(x, (784, 1)) for x in dt]


biases=[]
weigths=[]
with open("parta_biases.pkl","r") as f:
    biases=pickle.load(f)
    #print type(var)
with open("parta_weights.pkl","r") as f:
    weights=pickle.load(f)
predict(ted,labels,biases,weights)
    
