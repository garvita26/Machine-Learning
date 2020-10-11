import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.manifold import TSNE
from sklearn import svm
import os
import os.path
import argparse
import itertools
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
import operator
import copy
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--classifier", type = str  )
args = parser.parse_args()



def ROC(ylabel,yscore):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in yscore:
        
        fpr, tpr, _ = metrics.roc_curve(ylabel, yscore[i])
        roc_auc = metrics.auc(fpr, tpr)

    print tpr 
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show() 

    

def decisionfun(vs,w,b):
    distance=[]
    for j in range(len(vs)):
            #print np.shape(weights[i])
        distance.append((np.dot(w.T,vs[j])+b)/np.linalg.norm(w))
    return distance           
        

        
def modified(i,j,ts,tl):
    mod_ts=[]
    mod_tl=[]
    for k in range(len(ts)):
        if tl[k]==i or tl[k]==j:
            mod_ts.append(ts[k])
            mod_tl.append(tl[k])
    return mod_ts,mod_tl        
        


def ovo(training_set, validation_set,training_labels, validation_labels):
   
    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    weights={}
    intercept={}
    sp={}
    y_score={}
    #print training_labels
    for i in range(0,len(s)-1):
        for j in range(i+1,len(s)):
            mod_ts,mod_tl=modified(i,j,training_set,training_labels)
            #mod_ts,mod_tl=modified(i,j,X,Y)
            tl=copy.deepcopy(mod_tl)
            
            #print "training label without modification ",tl
            for k in range(len(mod_tl)):
                if(mod_tl[k])!=s[i]:
                    tl[k]=-1
                else:
                    tl[k]=1
            #print "training label after modification ",tl  

            clf=SVC(kernel='linear').fit(training_set,training_labels)
            weights[str(i)+" "+str(j)]=clf.coef_[0]
            intercept[str(i)+" "+str(j)]=clf.intercept_[0]
            sp[str(i)+" "+str(j)]=clf.support_vectors_
            y_score[str(i)+" "+str(j)] = decisionfun(validation_set,weights[str(i)+" "+str(j)],intercept[str(i)+" "+str(j)])
    print "yscore ",y_score        
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
   
    ROC(validation_labels,y_score)
    


def ovr(training_set, validation_set,training_labels, validation_labels):

    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    weights={}
    intercept={}
    sp={}
    y_score={}
    clf=SVC(kernel='linear').fit(training_set,training_labels)
    weights[0]=clf.coef_[0]
    intercept[0]=clf.intercept_[0]
    sp[0]=clf.support_vectors_
    y_score[0] = decisionfun(validation_set,weights[0],intercept[0])
    #print training_labels

    
        
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
                #print "hi"
                temp=[] 
                for l in s:
                    if l!=i:
                        #print l,i
                        temp.append(l)
                temp.append(np.dot(weights[i].T,validation_set[j])+intercept[i])
                #print "temp ",temp
                predict.append(temp)
        #print "pre ",predict        
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
    
    ROC(validation_labels,y_score)
    

def accuracy(predictions_final,validation_labels):
    count=0
    for i in range(len(predictions_final)):
        if predictions_final[i]==validation_labels[i]:
            count=count+1
    return(float(count)/len(validation_labels))


path='/home/garvita/Desktop/ml/hw/Template/Data/part_C_train.h5'
#path='/home/garvita/Desktop/Assignment2/data_5.h5'
f=h5py.File(path, "r+")
print path
#print f
#print "keys: %s" %f.keys()
#dataset=f['x'][:]
#y_data=f['y'][:]

#for previous assignment files
d=f['X']
dataset=TSNE(n_components=2).fit_transform(d) 
#print dataset
label=f['Y'][:]
y_data=[]
k=0
for i in range(len(label)):
    for j in range(len(label[0])):
        if(label[i][j]==1):
            y_data.append(j) 

size = np.array(dataset).shape[0]
td=dataset[0:(size*4)/5]
tl=y_data[0:(size*4)/5]
vd=dataset[(size*4)/5:]
vl=y_data[(size*4)/5:]

if args.classifier=="ovo":
    ovo(td, vd,tl, vl)
elif args.classifier=="ovr":
    ovr(td, vd,tl, vl)


    

        
