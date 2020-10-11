from __future__ import division
import numpy as np


# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):
        total_mean={}
        total_var={}
        all_cls={}
        prior={}
        
        
        
        
	def __init__(self ):
		# define all the model weights and state here
		pass
        
                
                        
                                
                
	def fit(self,X,Y):
                sd={}
                for i in set(Y):
                        sd[i]=[]
                for i in range(len(X)):
                        m=sd[Y[i]]
                        m.append(X[i])
                        sd[Y[i]]=m
                
                #print sd
                #classes in Y
               
                
                #print self.all_cls
                count_sample=np.array(X).shape[0]
                #calculated prior probability of each class
                
                for i in sd:
                        self.prior[i]=len(sd[i]) / count_sample
                #class_log_prior_ = [np.log(len(i) / count_sample) for i in sd]
                #calculated mean and variance of each feature of each class
                temp=[]
                mean_attribute=[]
                var_attribute=[]
                
                for i in sd:
                        #print i
                        classes=sd[i]
                        mean_attribute=[]
                        var_attribute=[]
                        for k in range(len(classes[0])):
                                temp=[]
                                for j in range(len(classes)):
                                        temp.append(classes[j][k])
                                mean_attribute.append(np.mean(temp))
                                var_attribute.append(np.var(temp))
                        self.total_mean[i]=mean_attribute
                        self.total_var[i]=var_attribute
                 
                 
               # print self.total_mean
                #print self.total_var
                for i in np.unique(Y):
                        self.all_cls[i]=0
                #print self.all_cls

	def predict(self,X ):
                #print self.all_cls
                #print self.prior
                output=[]
                for i in range(len(X)):
                        posterior_class={}
                        for cls in self.all_cls:
                                #print cls
                                p_all_features=[]
                                for j in range(len(X[i])):
                                        mean=self.total_mean[cls]
                                        variance=self.total_var[cls]
                                        #print variance
                                        #print mean
                                        if variance[j]!=0 :
                                                p=1/(np.sqrt(2*np.pi*variance[j])) * np.exp((-(X[i][j]-mean[j])**2)/(2*variance[j]))
                                        else:
                                                p=1
                                        p_all_features.append(np.log(p))  
                                #print(p_all_features)        
                                posterior_class[cls]=(self.prior[cls]*np.sum(p_all_features))
        
                        #print posterior_class
                        temp=[]
                        for i in posterior_class:
                                temp.append(posterior_class[i])
                        temp2=np.max(temp)    
                        y=dict((v,k) for k, v in posterior_class.iteritems())
                        output.append(y[temp2])
                return output 
                        
	def score(self,X,Y):
                count=0
                predicted=GaussianNB().predict(X)
                for i in range(len(Y)):
                        if predicted[i]==Y[i]:
                                count=count+1
                return(float(count)/len(Y))                
                                
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
#GaussianNB().fit(X,Y)
#z=GaussianNB().predict([[-0.8, -1]])        
#print z
