import h5py
import numpy as np
from mnist import MNIST
from sklearn.model_selection import KFold
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train(training_data,training_labels, epochs,eta,test_data,test_labels):
    clf = MLPClassifier(activation='logistic',learning_rate_init=eta,learning_rate='adaptive',max_iter=epochs,hidden_layer_sizes=(1000,500,200),verbose=1)
    with open("Q3bstr3_model.pkl","wb") as f:
        clf.fit(training_data,training_labels)
        pickle.dump(clf,f)
        predict_labels=clf.predict(test_data)
        #print "predicted ",predict_labels
        score=evaluate(test_labels,predict_labels)
        print "score ",score    
          






def evaluate(test_labels,predict_labels):
    count=0
    for i in range(len(test_labels)):
        if(predict_labels[i]==test_labels[i]):
            #print "hi"
            count=count+1
                    
    #print count                
    return (float(count)/len(test_labels))



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
test=[]
for i in range(len(te_d)):
    test.append(te_d[i])
    


accuracy_allfold=[]
biases_all=[]
weights_all=[]
train(dataset, labels,200, .0001,test,te_l)




        





#biases,weights=SGD(training_inputs, training_labels,50, len(training_inputs), .00001,biases,weights,num_layers,test_inputs,te_l)

