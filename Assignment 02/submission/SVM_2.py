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
from sklearn.svm import SVC
import operator
import copy
parser = argparse.ArgumentParser()
parser.add_argument("--classifier", type = str  )
args = parser.parse_args()


def modified(i,j,ts,tl):
    mod_ts=[]
    mod_tl=[]
    for k in range(len(ts)):
        if tl[k]==i or tl[k]==j:
            mod_ts.append(ts[k])
            mod_tl.append(tl[k])
    return mod_ts,mod_tl

def ovo(training_set, validation_set,training_labels, validation_labels,param):
   
    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    dc={}
    ind_support={}
    intercept={}
    allsupport={}
    #print training_labels
    for i in range(0,len(s)-1):
        for j in range(i+1,len(s)):
           
            mod_ts,mod_tl=modified(i,j,training_set,training_labels)
            tl=copy.deepcopy(mod_tl)
            
            #print "training label without modification ",tl
            for k in range(len(mod_tl)):
                if(mod_tl[k])!=s[i]:
                    tl[k]=-1
                else:
                    tl[k]=1
            #print "training label after modification ",tl  
            clf=SVC(kernel='rbf',C=param[0],max_iter=param[1],tol=param[2]).fit(mod_ts,tl)
            dc[str(i)+" "+str(j)]=clf.dual_coef_[0]
            ind_support[str(i)+" "+str(j)]=clf.support_[0]
            allsupport[str(i)+" "+str(j)]=clf.support_vectors_
            intercept[str(i)+" "+str(j)]=clf.intercept_[0]
    #print "indices ",ind_support
    #print "support vectors  ",allsupport
    #print "dual coeff ",dc
    predict_all={}
    weights={}
    for i in allsupport:
        weights[i]=np.dot(dc[i],allsupport[i])
        
        
    
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
    #print "\n" 
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
    
    score=accuracy(validation_labels,predictions_final)
    return score


def ovr(training_set, validation_set,training_labels, validation_labels,param):

    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    dc={}
    ind_support={}
    intercept={}
    allsupport={}
    #print training_labels
    for i in range(0,len(s)):
        tl=copy.deepcopy(training_labels)
        #print "training label without modification ",tl
        for k in range(len(training_labels)):
            if(training_labels[k])!=s[i]:
                tl[k]=-1
            else:
                tl[k]=1
            #print "training label after modification ",tl  
            clf=SVC(kernel='rbf',C=param[0],max_iter=param[1],tol=param[2]).fit(training_set,tl)
            dc[i]=clf.dual_coef_[0]
            ind_support[i]=clf.support_[0]
            allsupport[i]=clf.support_vectors_
            intercept[i]=clf.intercept_[0]
    #print weights
    #print intercept
   
    predict_all={}
    weights={}
    for i in allsupport:
        weights[i]=np.dot(dc[i],allsupport[i])
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
    score=accuracy(predictions_final,validation_labels)
    return score

def accuracy(predictions_final,validation_labels):
    count=0
    for i in range(len(predictions_final)):
        if predictions_final[i]==validation_labels[i]:
            count=count+1
    return(float(count)/len(validation_labels))

def partition(X,fold,K):
    size = np.array(X).shape[0]
    start = (size/K)*fold
    end = (size/K)*(fold+1)
    validation=X[start:end]
    training = np.concatenate((X[:start], X[end:]))
    return training,validation
                               
def cross_validation(classifier,k,data,labels,param):
    validation_folds_score = []
    #validation_folds_score = []
    for fold in range(0, k):
        training_set, validation_set = partition(data, fold,k)
        training_labels, validation_labels =partition(labels,fold,k)
        if classifier=="ovo":
            score=ovo(training_set, validation_set,training_labels, validation_labels,param)
            #validation_folds_score.append(v)
        else:
            score=ovr(training_set, validation_set,training_labels, validation_labels,param)
            
        validation_folds_score.append(score)
        #model.fit(training_set, training_labels)
        #validation_folds_score.append(model.score(validation_set, validation_labels))
                                       
        #training_predicted = model.predict(training_set)
        #validation_predicted = model.predict(validation_set)
        #train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
        #validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
        #return train_folds_score, validation_folds_score
    return validation_folds_score
def grid_search(grid):
    g=[]
    m=[]
    for i in grid.keys():
        
        g.append(grid[i])
    m=list(itertools.product(*g))
    return m    
    
    


#path='/home/garvita/Desktop/ml/hw/Template/Data/part_C_train.h5'
path='/home/garvita/Desktop/Assignment2/data_1.h5'
f=h5py.File(path, "r+")
print path
print f
#print "keys: %s" %f.keys(2
dataset=f['x'][:]
y_data=f['y'][:]

#for previous assignment files
'''d=f['X']
dataset=TSNE(n_components=2).fit_transform(d) 
#print dataset
label=f['Y'][:]
y_data=[]
k=0
for i in range(len(label)):
    for j in range(len(label[0])):
        if(label[i][j]==1):
            y_data.append(j) '''  
grid = {'C': [1],'max_iter': [-1],'tol' : [.001]}
#print type(grid)
m=grid_search(grid)
#print len(m)

scores=[]
for i in list(m):
    if args.classifier=="ovo":
        s=cross_validation("ovo",5,dataset,y_data,i)
    elif args.classifier=="ovr":
        s=cross_validation("ovr",5,dataset,y_data,i)
    avg_score=np.mean(s)
    scores.append(avg_score)
m_scores=max(scores)        
for i in list(scores):
    #print scores
    if(i==m_scores):
        ind=scores.index(i)
        break
#print scores
print m[ind]
print m_scores


    

        
