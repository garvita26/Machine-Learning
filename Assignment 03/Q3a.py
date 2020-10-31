import h5py
import numpy as np
from sklearn.model_selection import KFold
import pickle
from sklearn.neural_network import MLPClassifier




def train(training_data,training_labels, epochs, eta,test_data,test_labels):
    print np.shape(test_data[0])
    #print training_labels
    n = len(training_data)
    accuracy_K=[]
    for j in xrange(1,epochs,10):
        clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=eta,learning_rate='adaptive',max_iter=j,hidden_layer_sizes=(200,100))
        clf.fit(training_data,training_labels)
        predict_labels=clf.predict(test_data)
        #print "predicted ",predict_labels
        score=evaluate(test_labels,predict_labels)
        print "Epoch {0}: {1}".format(j,score)
        temp=[]
        temp.append(j)
        temp.append(score)
        accuracy_K.append(temp)
    #else:
     #   print "Epoch {0} complete".format(j)    
        
    return accuracy_K    





def evaluate(test_labels,predict_labels):
    count=0
    for i in range(len(test_labels)):
        
        if(predict_labels[i]==test_labels[i]):
            #print "hi"
            count=count+1
                    
    #print count                
    return (float(count)/len(test_labels))



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
    if i==7:
        l.append(0)
    else:
        l.append(1)
print "labels ",np.shape(np.array(labels))        
l=np.array(l)
print "l ",np.shape(l)
accuracy_allfold=[]

X_train, X_test = dt[:(4*len(dt))/5], dt[(4*len(dt))/5:]
y_train, y_test = l[:(4*len(dt))/5], l[(4*len(dt))/5:]
#td=[np.reshape(x, (784, 1)) for x in X_train]
#ted=[np.reshape(x, (784, 1)) for x in X_test]
print "train ",np.shape(y_train)
print "test ",np.shape(y_test)
accuracy_K=train(X_train, y_train,100, .0001,X_test,y_test)
accuracy_allfold.append(accuracy_K)
    

index=0
for i in range(len(accuracy_allfold)):
    max_val=0
    index=0
    if accuracy_allfold[i][-1][1]>max_val:
        max_val=accuracy_allfold[i][-1][1]
        index=i

#print "index ",index
clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=.0001,learning_rate='adaptive',max_iter=100,hidden_layer_sizes=(200,100))
        
with open("Q3astr3_model.pkl","wb") as f:
    clf.fit(dt,l)
    pickle.dump(clf,f)


with open("epoch_accuracy_Q3astr3.txt","w") as f:
    for i in accuracy_allfold:
        f.write(str(i))
        f.write("\n")

