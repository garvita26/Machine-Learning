import h5py
import numpy as np
from mnist import MNIST
from sklearn.model_selection import KFold
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train(training_data,training_labels, epochs,eta,test_data,test_labels):
    
    #print training_labels
    n = len(training_data)
    accuracy_K=[]
    for j in xrange(1,epochs,10):
        clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=eta,learning_rate='adaptive',early_stopping=True,max_iter=j,hidden_layer_sizes=(100,50))
        clf.fit(training_data,training_labels)
        if j%10==0:
            predict_labels=clf.predict(test_data)
            #print "predicted ",predict_labels
            score=evaluate(test_labels,predict_labels)
        
            print "Epoch {0}:{1}".format(j, score)
            temp=[]
            temp.append(j)
            temp.append(score)
            accuracy_K.append(temp)
           
        
    return accuracy_K    






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

test_inputs = [np.reshape(x, (784, 1)) for x in te_d]
#test_data = zip(test_inputs, te_l)

kf = KFold(n_splits=5)
accuracy_allfold=[]
biases_all=[]
weights_all=[]
for train_index, test_index in kf.split(dataset):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = dataset[train_index], dataset[test_index]
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    y_train, y_test = labels[train_index], labels[test_index]
    #X_test = scaler.transform(X_test)
    
    accuracy_K=train(X_train, y_train,50, .0001,X_test,y_test)
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
        
clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=.001,learning_rate='adaptive',max_iter=50,hidden_layer_sizes=(100,50))
        
with open("Q2b_model.pkl","wb") as f:
    clf.fit(dataset,labels)
    pickle.dump(clf,f)


with open("epoch_accuracy_Q2b.txt","w") as f:
    for i in accuracy_allfold:
        f.write(str(i))
        f.write("\n")

#biases,weights=SGD(training_inputs, training_labels,50, len(training_inputs), .00001,biases,weights,num_layers,test_inputs,te_l)

