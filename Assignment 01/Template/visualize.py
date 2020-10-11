import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
from matplotlib import figure

from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()



def Part_A():
    f=h5py.File(args.data, "r+")
    dataset=f['X']
    #print dataset
    X_embedded = TSNE(n_components=2).fit_transform(dataset)
    #print X_embedded
    y_data=f['Y'][:]
    label=[]
    k=0
    for i in range(len(y_data)):
        for j in range(len(y_data[0])):
            if(y_data[i][j]==1):
                label.append(j)
            
    #print label
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(args.plots_save_dir)
    plt.show()
    
def Part_B():
    f=h5py.File(args.data, "r+")
    dataset=f['X']
    #print dataset
    X_embedded = TSNE(n_components=2).fit_transform(dataset)
    #print X_embedded
    y_data=f['Y'][:]
    label=[]
    k=0
    for i in range(len(y_data)):
        for j in range(len(y_data[0])):
            if(y_data[i][j]==1):
                label.append(j)
            
    #print label
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(args.plots_save_dir)
    plt.show()

def Part_C():
    f=h5py.File(args.data, "r+")
    dataset=f['X']
   # print dataset
    X_embedded = TSNE(n_components=2).fit_transform(dataset)
    #print X_embedded
    y_data=f['Y'][:]
    label=[]
    k=0
    for i in range(len(y_data)):
        for j in range(len(y_data[0])):
            if(y_data[i][j]==1):
                label.append(j)
            
    #print label
    plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(args.plots_save_dir)

if args.data=="Data/part_A_train.h5":
    Part_A()
elif  args.data=="Data/part_B_train.h5":
    Part_B()
elif args.data=="Data/part_C_train.h5":
    Part_C()
else:
        raise Exception("Invalid file name")
    
