import os
import os.path
import argparse
from sklearn.naive_bayes import GaussianNB
#from Models.GaussianNB import GaussianNB
from sklearn.linear_model import LogisticRegression
#from Models.LogisticRegression import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import h5py
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# Preprocess data and split it
def preprocess():
        X,Y=load_h5py(args.train_data)
        label=[]
        k=0
        for i in range(len(Y)):
                for j in range(len(Y[0])):
                        if(Y[i][j]==1):
                                label.append(j)
        return X,label
        

def partition(X,fold,K):
        size = np.array(X).shape[0]
        start = (size/K)*fold
        end = (size/K)*(fold+1)
        validation=X[start:end]
        training = np.concatenate((X[:start], X[end:]))
        return training,validation
                               
def cross_validation(model,k,data,labels):
                               validation_folds_score = []
                               #validation_folds_score = []
                               for fold in range(0, k):
                                       training_set, validation_set = partition(data, fold,k)
                                       training_labels, validation_labels =partition(labels,fold,k)
                                       model.fit(training_set, training_labels)
                                       validation_folds_score.append(model.score(validation_set, validation_labels))
                                       
                                       #training_predicted = model.predict(training_set)
                                       #validation_predicted = model.predict(validation_set)
                                       #train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
                                       #validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
                               #return train_folds_score, validation_folds_score
                               return validation_folds_score
def grid_search(grid):
        size=len(grid)
        l=[]
        m=[]
        for d in grid:
                for i in d:
                        l.append(d[i])
        #print l
        for i in l[0]:
                for j in l[1]:
                        for k in l[2]:
                                 x=[i,j,k]
                                 m.append(x)
                                 #m.append()
        return m
# Train the models

if args.model_name == 'GaussianNB':
                               X,Y=preprocess()
                               score1=cross_validation(GaussianNB(),10,X,Y)
                               #print score1
                               print np.mean(score1)
                               #print np.mean(score2)

                               model1=GaussianNB()
                               with open(args.weights_path+".pk","wb") as f:
                                       model1.fit(X,Y)
                                       pickle.dump(model1,f)
	                      # with open(args.weights_path+".pk","rb") as f:
                                      # var=pickle.load(f)
                                       #print var 
                            
elif args.model_name == 'LogisticRegression':
        grid = [{'C': [10**-4,10**-2, 10**0]},{'max_iter': [10,20]},{'penalty' : ['l1','l2']}]
        m=grid_search(grid)
        #print np.array(m).shape
        X,Y=preprocess()
        scores=[]
        
        for i in list(m):
                
                score1=cross_validation(LogisticRegression(C=i[0],max_iter=i[1],penalty=i[2]),10,X,Y)
                avg_score=np.mean(score1)
                scores.append(avg_score)
        m_scores=max(scores)        
        for i in list(scores):
                #print scores
                if(i==m_scores):
                        ind=scores.index(i)
                        break
        print scores
        print m[ind]
        print m_scores


        model1=LogisticRegression(C=m[ind][0],max_iter=m[ind][1],penalty=m[ind][2])
        with open(args.weights_path+".pk","wb") as f:
                model1.fit(X,Y)
                pickle.dump(model1,f)
       # with open(args.weights_path+".pk","rb") as f:
        #        var=pickle.load(f)
         #       print var
        x=[]
        for i in range(len(m)):
                x.append(i)
        X=[]        
        for i in m:
                 X.append(i)        
        #pylab.rcParams['xtick.major.pad']='20'        
        my_yticks=X
        plt.yticks(x,my_yticks)
        plt.plot(scores,x)
        plt.xlabel("validation accuracy")
        plt.ylabel("Parameters [C,max-iter,penalty]")
        plt.title("validation accuracy vs Parameter graph")
        plt.savefig(args.plots_save_dir)
        plt.show()

elif args.model_name == 'DecisionTreeClassifier':
         grid = [{'min_samples_leaf': [1,2,3]},{'max_depth': [None,1,10]},{'min_samples_split': [2,3,4,5]}]
         m=grid_search(grid)
         print np.array(m).shape
         X,Y=preprocess()
         scores=[]
         for i in list(m):
                 score1=cross_validation(DecisionTreeClassifier(min_samples_leaf=i[0],max_depth=i[1],min_samples_split=i[2]),10,X,Y)
                 avg_score=np.mean(score1) 
                 scores.append(avg_score)
         m_scores=max(scores)        
         for i in list(scores):
                 if(i==m_scores):
                         ind=scores.index(i)
                         break
         print m[ind]
         print m_scores
         model1=DecisionTreeClassifier(min_samples_leaf=m[ind][0],max_depth=m[ind][1],min_samples_split=m[ind][2])
         with open(args.weights_path+".pk","wb") as f:
                 model1.fit(X,Y)
                 pickle.dump(model1,f)
        # with open(args.weights_path+".pk","rb") as f:
         #       var=pickle.load(f)
          #      print var        
         x=[]
         for i in range(len(m)):
                 x.append(i)
         X=[]        
         for i in m:
                 X.append(i)
                 
                         
         #pylab.rcParams['xtick.major.pad']='20'        
         my_yticks=X
         plt.yticks(x,my_yticks)
         plt.plot(scores,x)
         plt.xlabel("validation accuracy")
         plt.ylabel("Parameters [min_samples_leaf,max_depth,min_samples_split]")
         plt.title("validation accuracy vs Parameter graph")
	 plt.savefig(args.plots_save_dir)	
         plt.show()
            
	
else:
        raise Exception("Invalid Model name")
