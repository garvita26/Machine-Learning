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

def decisionboundary(classifier,X,y):
    
    
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10,s=30, cmap=plt.cm.Paired,edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = min(X[:,0]) - 1, max(X[:,0]) + 1
    y_min, y_max = min(X[:,1]) - 1, max(X[:,1]) + 1
    print x_min,x_max,y_min,y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    xy=np.c_[xx.ravel(), yy.ravel()]
    if classifier=="ovo":
        Ztest,Z,s=ovo(X,y,xy)
    else:
        Ztest,Z,s=ovr(X,y,xy)
    x1=[]
    x2=[]
    for i in s:
        temp1=[]
        temp2=[]
        
        for j in s:
            #print j
            temp1.append(j[0])
            temp2.append(j[1])
        x1.append(temp1)
        x2.append(temp2)
    #print xy
    print "from built in ",np.shape(Ztest)
    print "from own ",np.shape(Z)
    for i in range(len(Z)):
        #print 'ztest ',Ztest[i]
        #print "Z ",Z[i]
        ax.contour(xx, yy, (np.array(Ztest[i])).reshape(xx.shape), colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=[ '--','-','--'])
        # plot support vectors
        ax.scatter(x1[i], x2[i], s=100,linewidth=1, facecolors='none')
        #plt.contour(xx, yy, (np.array(predictions[i])).reshape(xx.shape), cmap=plt.cm.Paired)
        #plt.scatter(x1[i], x2[i], s=80, facecolors='none', zorder=10, edgecolors='k')
    plt.show()

   

    
def modified(i,j,ts,tl):
    mod_ts=[]
    mod_tl=[]
    for k in range(len(ts)):
        if tl[k]==i or tl[k]==j:
            mod_ts.append(ts[k])
            mod_tl.append(tl[k])
    return mod_ts,mod_tl

def actual(training_set, validation_set,training_labels, validation_labels):
    clf=SVC(kernel='rbf')
    clf.fit(training_set,training_labels)
    p=clf.predict(validation_set)
    return p

def ovo(training_set,training_labels,ts):
   
    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    dc={}
    ind_support={}
    intercept={}
    allsupport={}
    Ztest=[]
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
            clf=SVC(kernel='rbf').fit(training_set,training_labels)
            Ztest.append(clf.predict(ts))
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
        for j in range(len(ts)):
            if np.dot(weights[i].T,ts[j])+intercept[i]>0:
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
    s=[]    
    for i in allsupport:
        s.append(allsupport[i])
    return Ztest,predictions,s    
    #print predictions



def ovr(training_set,training_labels,ts):

    #print "unique labels ",np.unique(training_labels)
    s=list(set(training_labels))
    #print "original ",training_labels
    
    #print s
    dc={}
    ind_support={}
    intercept={}
    allsupport={}
    Ztest=[]
    print "len ",len(s)
    
    for i in range(0,len(s)):
        for i in range(0,len(s)):
            tl=copy.deepcopy(training_labels)
            #print "training label without modification ",tl
            for k in range(len(training_labels)):
                if(training_labels[k])!=s[i]:
                    tl[k]=-1
                else:
                    tl[k]=1
                #print "training label after modification ",tl  
                clf=SVC(kernel='rbf').fit(training_set,training_labels)
                Ztest.append(clf.predict(ts))
                dc[i]=clf.dual_coef_[0]
                ind_support[i]=clf.support_[0]
                allsupport[i]=clf.support_vectors_
                intercept[i]=clf.intercept_[0]
    else:
        print 'hi'
        clf=SVC(kernel='linear').fit(training_set,training_labels)
        Ztest.append(clf.predict(ts))
        dc[0]=clf.dual_coef_[0]
        ind_support[0]=clf.support_[0]
        allsupport[0]=clf.support_vectors_
        intercept[0]=clf.intercept_[0]
    #print training_labels
    
    #print weights
    #print intercept
   
    predict_all={}
    weights={}
    for i in allsupport:
        weights[i]=np.dot(dc[i],allsupport[i])
    #print "testing ",validation_labels
    for i in weights:
        predict=[]
        for j in range(len(ts)):
            if np.dot(weights[i].T,ts[j])+intercept[i]>0:
                predict.append(1)
            else:
                predict.append(-1)
        predict_all[i]=predict    
    
    #print predict_all    
    
    predictions=[]
    for i in predict_all:
        predictions.append(predict_all[i])

    s=[]    
    for i in allsupport:
        s.append(allsupport[i])
        
    return Ztest,predictions,s     
    #print predictions

    

def accuracy(predictions_final,validation_labels):
    count=0
    for i in range(len(predictions_final)):
        if predictions_final[i]==validation_labels[i]:
            count=count+1
    return(float(count)/len(validation_labels))





f=h5py.File('/home/garvita/Desktop/Assignment2/data_2.h5', "r+")
#print f
#print "keys: %s" %f.keys()
dataset=f['x'][:]
y_data=f['y'][:]
    
size = np.array(dataset).shape[0]
td=dataset[0:(size*4)/5]
tl=y_data[0:(size*4)/5]
vd=dataset[(size*4)/5:]
vl=y_data[(size*4)/5:]

if args.classifier=="ovo":
    decisionboundary("ovo",dataset,y_data)
    #s,dc,allsupport,intercept=ovo(td, vd,tl, vl)
elif args.classifier=="ovr":
    decisionboundary("ovr",dataset,y_data)
    
    
#print s
#decisionboundary(dataset,y_data,dc,allsupport,intercept)


        
