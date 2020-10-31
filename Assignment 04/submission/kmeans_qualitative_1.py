import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from sklearn.manifold import TSNE


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
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.title("seeds dataset")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=Y)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    plt.show()
    
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
            
    X_embedded = TSNE(n_components=2).fit_transform(X)
    #print X_embedded
    plt.title("vertebral column")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    plt.show()
    
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
    #print Y        
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
            
    X_embedded = TSNE(n_components=2).fit_transform(X)
    #print label
    #print X_embedded
    plt.title("segmentation data")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    plt.show()


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
    #print Y        
    label=[]        
    for i in Y:
        if i=='Iris-setosa':
            label.append(1)
        elif i =='Iris-versicolor':
            label.append(2)
        elif i=='Iris-virginica':
            label.append(3)
        
            
    X_embedded = TSNE(n_components=2).fit_transform(X)
    #print label
    #print X_embedded
    plt.title("iris data")
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(args.plots_save_dir)
    plt.show()
    
#seedsdataset()            
#vertribalcolumn()
#segmentationdata()
irisdata()    
