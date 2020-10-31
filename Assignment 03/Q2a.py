import h5py
import numpy as np
from sklearn.model_selection import KFold
import pickle
from sklearn.neural_network import MLPClassifier
import itertools



def train(training_data,training_labels, epochs, eta,test_data,test_labels):
    print np.shape(test_data[0])
    #print training_labels
    n = len(training_data)
    accuracy_K=[]
    for j in xrange(1,epochs):
        clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=eta,learning_rate='adaptive',max_iter=j,hidden_layer_sizes=(100,50))
        clf.fit(training_data,training_labels)
        predict_labels=clf.predict(test_data)
        #print "predicted ",predict_labels
        score=evaluate(test_labels,predict_labels)
        print "Epoch {0}: {1}".format(
                    j,score)
        temp=[]
        temp.append(j)
        temp.append(score)
        accuracy_K.append(temp)
    #else:
     #   print "Epoch {0} complete".format(j)    
        
    return accuracy_K    


def grid_search(grid):
    g=[]
    m=[]
    for i in grid.keys():
        
        g.append(grid[i])
    m=list(itertools.product(*g))
    return m   



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
grid = {'batch_size': [10,20],'max_iter': [10,50,100],'eta' : [.1,.01,.001]}
m=grid_search(grid)
kf = KFold(n_splits=5)
accuracy_allfold=[]

for i in list(m):
    for train_index, test_index in kf.split(dt):
        X_train, X_test = dt[train_index], dt[test_index]
        y_train, y_test = l[train_index], l[test_index]
        #td=[np.reshape(x, (784, 1)) for x in X_train]
        #ted=[np.reshape(x, (784, 1)) for x in X_test]
        print "train ",np.shape(y_train)
        print "test ",np.shape(y_test)
        accuracy_K=train(X_train, y_train,i[1],i[2],X_test,y_test)
        accuracy_allfold.append(accuracy_K)
    

index=0
for i in range(len(accuracy_allfold)):
    max_val=0
    index=0
    if accuracy_allfold[i][-1][1]>max_val:
        max_val=accuracy_allfold[i][-1][1]
        index=i

#print "index ",index
clf = MLPClassifier(activation='logistic',batch_size=20,learning_rate_init=.001,learning_rate='adaptive',max_iter=50,hidden_layer_sizes=(100,50))
        
with open("Q2a_model.pkl","wb") as f:
    clf.fit(dt,l)
    pickle.dump(clf,f)


with open("epoch_accuracy_Q2a.txt","w") as f:
    for i in accuracy_allfold:
        f.write(str(i))
        f.write("\n")

