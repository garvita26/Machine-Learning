import numpy as np


# make sure this class id compatable with sklearn's LogisticRegression

class LogisticRegression(object):
        weight={}

	def __init__(self, penalty='l2' , C=1.0 , tol=.0001,max_iter=100 , verbose=0):
		self.penalty=penalty
		self.C=C
		self.max_iter=max_iter
		self.verbose=verbose
		self.tol=tol
        

        
        
	def fit(self,X , Y):
                lr=self.C
                #print lr
		X = np.insert(X, 0, 1, axis=1)
		#self.weight = {}
                m = len(X)
                for i in np.unique(Y):
                        y_copy = np.where(Y == i, 1, 0)
                        w = np.zeros(len(X[0]))
                        for j in range(self.max_iter):
                                    scores=np.dot(X, w)
                                    predictions=1/(1+np.exp(-scores))
                                    #tuning parameters using gradient descent
                                    # using formula gd(ll)=X.T(Y-predictions)
                                    error = y_copy- predictions
                                    gd = np.dot(X.T, error)
                                    w+=lr*gd
                                    #print w
                        self.weight[i]=w
                #print self.weight        
                return self
                                    
        def predict(self,X):
                output=[]
                X=np.insert(X,0,1,axis=1)
               # print self.weight
                for i in X:
                        temp={}
                        x=0
                        for c in self.weight:
                                temp[c]=i.dot(self.weight[c])
                        #temp2=np.max(temp)
                        #print temp
                        #print temp2
                        l=[]
                        for key in temp:
                                l.append(temp[key])
                        #print l
                        y=max(l)
                        #print y
                        for key in temp:
                                if temp[key]==y:
                                        x=key
                        output.append(x)
                #print output        
                return output                    
                
        def score(self,X,Y):
                count=0
                predicted=LogisticRegression().predict(X)
                for i in range(len(Y)):
                        if predicted[i]==Y[i]:
                                count=count+1
                return(float(count)/len(Y))                                     
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
#LogisticRegression().fit(X,Y)
#z=LogisticRegression().predict([[-.8,-.1],[-.8,-2]])
#print z
