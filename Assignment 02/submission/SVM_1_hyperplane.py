import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.manifold import TSNE
from sklearn import svm
import os
import os.path
import argparse
import itertools

from sklearn import metrics
from sklearn.svm import SVC
import operator
import copy


parser = argparse.ArgumentParser()
parser.add_argument("--classifier", type = str  )
args = parser.parse_args()

def hyperplane(X,y,weights,intercept,sp):
    #print "weights is ",weights
    #print "sp is ",sp
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    #print "xy is ",xy
    distance_all={}
    print "len of weights ",len(weights) 
    for i in weights:
        distance=[]
        for j in range(len(xy)):
            #print np.shape(weights[i])
            
            distance.append((np.dot(weights[i].T,xy[j])+intercept[i])/np.linalg.norm(weights[i]))
               
        distance_all[i]=distance
    d=[]
    for i in distance_all:
        d.append(np.array(distance_all[i]).reshape(XX.shape))
    
    s=[]
    x1=[]
    x2=[]
    for i in sp:
        temp1=[]
        temp2=[]
        s.append(sp[i])
        for j in sp[i]:
            #print j
            temp1.append(j[0])
            temp2.append(j[1])
        x1.append(temp1)
        x2.append(temp2)
            
            
    #print "x1 is ",x1[0]
    #print "x2 is " ,x2[0]
    #print len(s)
    # plot decision boundary and margins
    for i in range(len(d)):
        print "hi"
        ax.contour(XX, YY, d[i], colors='k', levels=[ -1,0,1],linestyles=['--', '-', '--'], alpha=0.5)
        # plot support vectors
        ax.scatter(x1[i], x2[i], s=100,linewidth=1, facecolors='none')
    plt.show()
     
def modified(i,j,ts,tl):
    mod_ts=[]
    mod_tl=[]
    for k in range(len(ts)):
        if tl[k]==i or tl[k]==j:
            mod_ts.append(ts[k])
            mod_tl.append(tl[k])
    return mod_ts,mod_tl        
        
        


def ovo(X,Y,training_set, validation_set,training_labels, validation_labels):
   
    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    weights={}
    intercept={}
    sp={}
    #print training_labels
    for i in range(0,len(s)-1):
        for j in range(i+1,len(s)):
            #mod_ts,mod_tl=modified(i,j,training_set,training_labels)
            mod_ts,mod_tl=modified(i,j,X,Y)
            tl=copy.deepcopy(mod_tl)
            
            #print "training label without modification ",tl
            for k in range(len(mod_tl)):
                if(mod_tl[k])!=s[i]:
                    tl[k]=-1
                else:
                    tl[k]=1
            #print "training label after modification ",tl  

            clf=SVC(kernel='linear',probability=True).fit(mod_ts,tl)
            weights[str(i)+" "+str(j)]=clf.coef_[0]
            intercept[str(i)+" "+str(j)]=clf.intercept_[0]
            sp[str(i)+" "+str(j)]=clf.support_vectors_
    #print weights
    #print intercept
    predict_all={}
    #print "testing ",validation_labels
    for i in weights:
        predict=[]
        for j in range(len(validation_set)):
            if np.dot(weights[i].T,validation_set[j])+intercept[i]>0:
                predict.append(1)
            else:
                predict.append(-1)
        predict_all[i]=predict
        
    #print predict_all
    for i in predict_all:
        label1= int(i.split(" ")[0])
        label2= int(i.split(" ")[1])
        for j in range(len(predict_all[i])):
            if(predict_all[i][j]==1):
                predict_all[i][j]=label1
            else:
                predict_all[i][j]=label2
     
    #print predict_all
    predictions=[]
    for i in predict_all:
        predictions.append(predict_all[i])
        
    #print predictions
    predictions_final=[]
    temp=zip(*predictions)
    #print temp
    for i in temp:
        #print i
        count={}
        for j in i:
            if j not in count:
                count[j]=1
            else:
                count[j]+=1
        #print count        
        predictions_final.append(max(count.iteritems(), key=operator.itemgetter(1))[0] )
    
    return weights,intercept,sp


def ovr(X,Y,training_set, validation_set,training_labels, validation_labels):

    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    weights={}
    intercept={}
    sp={}
    #print training_labels

    if len(s)>2:
        for i in range(0,len(s)):
            tl=copy.deepcopy(Y)
            #print "training label without modification ",tl
            for k in range(len(Y)):
                if(Y[k])!=s[i]:
                    tl[k]=-1
                else:
                    tl[k]=1
            #print "training label after modification ",tl  
            clf=SVC(kernel='linear').fit(X,tl)
            weights[i]=clf.coef_[0]
            intercept[i]=clf.intercept_[0]
            sp[i]=clf.support_vectors_
    else:
        clf=SVC(kernel='linear').fit(X,Y)
        weights[0]=clf.coef_[0]
        intercept[0]=clf.intercept_[0]
        sp[0]=clf.support_vectors_
        
    #print weights
    #print intercept
    predict_all={}
    #print "testing ",validation_labels
    for i in weights:
        predict=[]
        for j in range(len(validation_set)):
            if np.dot(weights[i].T,validation_set[j])+intercept[i]>0:
                predict.append([i,np.dot(weights[i].T,validation_set[j])+intercept[i]])
            else:
                temp=[] 
                for l in s:
                    if l!=i:
                        temp.append(l)
                    temp.append(np.dot(weights[i].T,validation_set[j])+intercept[i])
                    predict.append(temp)
        predict_all[i]=predict
    #print predict_all    
    
    predictions=[]
    for i in predict_all:
        predictions.append(predict_all[i])
        
    #print predictions
    predictions_final=[]
    temp=zip(*predictions)
    #print temp
    for i in temp:
        #print "tuple is ",i 
        count={}
        for j in i:
            #print "value inside tuple ",type(j)
            for k in range(len(j)-1):
                if j[k] not in count:
                    #print " class is ",j[k]
                    count[j[k]]=1
                else:
                    count[j[k]]+=1
            #print count        
        predictions_final.append(max(count.iteritems(), key=operator.itemgetter(1))[0] )
   
    return weights,intercept,sp

def accuracy(predictions_final,validation_labels):
    count=0
    for i in range(len(predictions_final)):
        if predictions_final[i]==validation_labels[i]:
            count=count+1
    return(float(count)/len(validation_labels))


#path='/home/garvita/Desktop/ml/hw/Template/Data/part_C_train.h5'
path='/home/garvita/Desktop/Assignment2/data_3.h5'
f=h5py.File(path, "r+")
print path
#print f
#print "keys: %s" %f.keys()
dataset=f['x'][:]
y_data=f['y'][:]

#for previous assignment files
'''dataset=f['X']
#print dataset
label=f['Y'][:]
y_data=[]
k=0
for i in range(len(label)):
    for j in range(len(label[0])):
        if(label[i][j]==1):
            y_data.append(j)
X_embedded = TSNE(n_components=2).fit_transform(dataset)   '''         

size = np.array(dataset).shape[0]
td=dataset[0:(size*4)/5]
tl=y_data[0:(size*4)/5]
vd=dataset[(size*4)/5:]
vl=y_data[(size*4)/5:]

if args.classifier=="ovo":
    weights,intercept,sp=ovo(dataset,y_data,td, vd,tl, vl)
elif args.classifier=="ovr":
    weights,intercept,sp=ovr(dataset,y_data,td, vd,tl, vl)

hyperplane(dataset,y_data,weights,intercept,sp)



    

        
