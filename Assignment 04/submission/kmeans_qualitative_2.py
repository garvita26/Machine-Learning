import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics
import random


def Scores(X,label,Xnew,predicted):
    p=[]
    for i in X:
        ind=Xnew.index(i)
        p.append(predicted[ind])
    ARI =metrics.adjusted_rand_score(label,p)
    NMI= metrics.normalized_mutual_info_score(label,p)
    AMI=metrics.adjusted_mutual_info_score(label,p)
    return ARI,NMI,AMI


def plotobjvsiter(o,title):
    X=[]
    for i in range(len(o)):
        X.append(i+1)
    plt.plot(X,o)
    plt.xlabel('iterations')
    plt.ylabel('objective function')
    plt.title(title)
    plt.show()
       
    
    

def randomcentroids(X,k):
    indexvalues=[]
    centroids=[]
    for i in range(len(X)):
        indexvalues.append(i)
    samplepoints=random.sample(set(indexvalues),k)
    for i in samplepoints:
        centroids.append(X[i])

    return centroids    
     
    
def comparedict(c1,c2):
    flag=0
    for i in range(len(c1)):
        if len(c1[i])==len(c2[i]):
            for j in range(len(c1[i])):
                if cmp(c1[i][j],c2[i][j])==0:
                    continue
                else:
                    return False
        else:
            return False
    return True    

def calculateRSS(C,centroids):
    RSS=0
    for i in C:
        RSS_k=0
        for j in C[i]:
            RSS_k=RSS_k+ np.linalg.norm(np.array(j)-np.array(centroids[i]))
        RSS=RSS+RSS_k
    return RSS    
            

def kmeans(X,k):
    #intial assignment
    Objective=[]
    centroids=randomcentroids(X,k)
    Clusters_prev={}
    Clusters_curr={}
    for i in range(k):
        Clusters_prev[i]=[]
        Clusters_curr[i]=[]
    
    #iteration 1    
    for n in range(len(X)):
        distance=[]
        for c in centroids:
            distance.append(np.linalg.norm(np.array(X[n])-np.array(c)))
        j=np.argmin(distance)
        Clusters_curr[j].append(X[n])
    for i in Clusters_curr:
        centroids[i]=np.mean(np.array(Clusters_curr[i]),axis=0)
    Objective.append(calculateRSS(Clusters_curr,centroids))    
    #print Clusters_curr    
    count = 2
    #iteration 2 to so on     
    while comparedict(Clusters_prev,Clusters_curr)==False:
    
        print "iteration number ",count
        count=count+1
        Clusters_prev.clear()
        Clusters_prev=Clusters_curr.copy()
        #print Clusters_prev
        Clusters_curr.clear()
        for i in range(k):
            Clusters_curr[i]=[]
        for n in range(len(X)):
            distance=[]
            for c in centroids:
                distance.append(np.linalg.norm(np.array(X[n])-np.array(c)))
            j=np.argmin(distance)
            Clusters_curr[j].append(X[n])
        for i in Clusters_curr:
            centroids[i]=np.mean(np.array(Clusters_curr[i]),axis=0)
        Objective.append(calculateRSS(Clusters_curr,centroids))     

    new_X=[]
    new_Y=[]
    #predicting labels
    for i in Clusters_curr:
        for j in Clusters_curr[i]:
            new_X.append(j)
            new_Y.append(i)

    return new_X,new_Y,Objective        
       
def seedsdataset():
    X=[]
    Y=[]
    with open("seeds_dataset.txt","r") as f:
        for lines in f:
            lines=lines.replace("\n","")
            values=lines.split("\t")
            Y.append(int(values[-1]))
            temp=[]
            for i in range(len(values)-1):
                try:
                    temp.append(float(values[i]))
                except:
                    pass
                
            X.append(temp)
    Xnew,predicted,o=kmeans(X,3)
    ARI,NMI,AMI=Scores(X,Y,Xnew,predicted)
    #plotobjvsiter(o,"seeds")
    X_embedded = TSNE(n_components=2).fit_transform(Xnew)
    plt.title("seeds dataset")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=predicted)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    #plt.show()
    return ARI,NMI,AMI

def vertribalcolumn():
    X=[]
    Y=[]
    with open("column_3C.dat","r") as f:
        for lines in f:
            lines=lines.replace("\n","")
            values=lines.split(" ")
            Y.append((values[-1]))
            temp=[]
            for i in range(len(values)-1):
                try:
                    temp.append(float(values[i]))
                except:
                    pass
                
            X.append(temp)
    label=[]        
    for i in Y:
        if i=='DH':
            label.append(1)
        elif i =='SL':
            label.append(2)
        else:
            label.append(3)        
    Xnew,predicted,o=kmeans(X,3) 
    #plotobjvsiter(o,"vertebral")
    ARI,NMI,AMI=Scores(X,label,Xnew,predicted)        
    X_embedded = TSNE(n_components=2).fit_transform(Xnew)
    plt.title("vertebral column")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=predicted)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    #plt.show()
    return ARI,NMI,AMI  


def segmentationdata():
    X=[]
    Y=[]
    with open("segmentation.data","r") as f:
        for lines in f:
            lines=lines.replace("\n","")
            values=lines.split(",")
            Y.append((values[:1]))
            temp=[]
            for i in range(1,len(values)):
                try:
                    temp.append(float(values[i]))
                except:
                    pass
                
            X.append(temp)
    label=[]        
    for i in Y:
        if i==['BRICKFACE']:
            label.append(1)
        elif i ==['SKY']:
            label.append(2)
        elif i==['FOLIAGE']:
            label.append(3)
        elif i==['CEMENT']:
            label.append(4)
        elif i==['WINDOW']:
            label.append(5)
        elif i==['PATH']:
            label.append(6)
        else:
            label.append(7)
    Xnew,predicted,o=kmeans(X,7)
    #plotobjvsiter(o,"segmentation"
    ARI,NMI,AMI= Scores(X,label,Xnew,predicted)   
    X_embedded = TSNE(n_components=2).fit_transform(Xnew)
    plt.title("segmentation data")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=predicted)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    #plt.show()
    return ARI,NMI,AMI


def irisdata():
    X=[]
    Y=[]
    with open("iris.data","r") as f:
        for lines in f:
            lines=lines.replace("\n","")
            values=lines.split(",")
            Y.append((values[-1]))
            temp=[]
            for i in range(len(values)-1):
                try:
                    temp.append(float(values[i]))
                except:
                    pass
                
            X.append(temp)
    label=[]        
    for i in Y:
        if i=='Iris-setosa':
            label.append(1)
        elif i =='Iris-versicolor':
            label.append(2)
        elif i=='Iris-virginica':
            label.append(3)
            
    Xnew,predicted,o=kmeans(X,3)
   # plotobjvsiter(o,"iris"
    ARI,NMI,AMI=Scores(X,label,Xnew,predicted)   
    X_embedded = TSNE(n_components=2).fit_transform(Xnew)
    plt.title("iris data")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=predicted)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    #plt.show()
    return ARI,NMI,AMI




valuesARI=[]
valuesNMI=[]
valuesAMI=[]

'''for i in range(5):
    ARI,NMI,AMI=vertribalcolumn()
    valuesARI.append(ARI)
    valuesNMI.append(NMI)
    valuesAMI.append(AMI)
print "ARI ",np.mean(valuesARI)
print "NMI ",np.mean(valuesNMI)
print "AMI ",np.mean(valuesAMI)'''

ARI,NMI,AMI=irisdata()
print ARI
print NMI
print AMI
#seedsdataset()            
#vertribalcolumn()
#segmentationdata()
#irisdata()    
