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




def confusionmatrix(vl,pl):
    #print "hi"
    s=list(set(vl))
    cm={}
    #print "vl is ",vl
    #print "pl is ",pl
    #cm = defaultdict(list)
    for i in range(len(s)):
        for j in range(len(s)):
            cm[str(i)+" "+str(j)]=0
            
    
   
    for j in range(len(vl)):
           cm[str(vl[j])+" "+str(pl[j])]+=1
    #print cm       
    confusion=np.zeros(shape=(len(s),len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            confusion[s[i],s[j]]=np.array(cm[str(s[i])+" "+str(s[j])])
    print confusion          
        
           
        
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

            clf=SVC(kernel='linear').fit(mod_ts,tl)
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
   
    confusionmatrix(validation_labels,predictions_final)
    


def ovr(training_set, validation_set,training_labels, validation_labels):

    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    weights={}
    intercept={}
    sp={}
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
        clf=SVC(kernel='linear').fit(training_set,tl)
        weights[i]=clf.coef_[0]
        intercept[i]=clf.intercept_[0]
        sp[i]=clf.support_vectors_
    
        
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
    
    confusionmatrix(validation_labels,predictions_final)
    

def accuracy(predictions_final,validation_labels):
    count=0
    for i in range(len(predictions_final)):
        if predictions_final[i]==validation_labels[i]:
            count=count+1
    return(float(count)/len(validation_labels))


path='/home/garvita/Desktop/ml/hw/Template/Data/part_B_train.h5'
#path='/home/garvita/Desktop/Assignment2/data_.h5'
f=h5py.File(path, "r+")
print path
#print f
#print "keys: %s" %f.keys()
#dataset=f['x'][:]
#y_data=f['y'][:]

#for previous assignment files
dataset=f['X']
#print dataset
label=f['Y'][:]
y_data=[]
k=0
for i in range(len(label)):
    for j in range(len(label[0])):
        if(label[i][j]==1):
            y_data.append(j)    
#grid = {'C': [10**-4,10**-2, 10**0],'max_iter': [-1,10,100],'tol' : [.001,.01,1]}
#print type(grid)
#m=grid_search(grid)
#print len(m)
size = np.array(dataset).shape[0]
td=dataset[0:(size*4)/5]
tl=y_data[0:(size*4)/5]
vd=dataset[(size*4)/5:]
vl=y_data[(size*4)/5:]

if args.classifier=="ovo":
    ovo(td, vd,tl, vl)
elif args.classifier=="ovr":
    ovr(td, vd,tl, vl)
    
#print s
#decisionboundary(dataset,y_data,weights,intercept,sp)



    

        
