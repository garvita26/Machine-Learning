import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure



parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()



def Part_A(file1,file2):
    f=h5py.File(file1, "r+")
    #print f
    #print "keys: %s" %f.keys()
    dataset=f['x'][:]
    print type(dataset)
    x=[]
    y=[]
    for i in dataset:
        x.append(i[0])
        y.append(i[1])
    #print x
    #print y
    threshold=1
    y_data=f['y'][:]
    seperated_dataset={}
    for i in range(len(dataset)):
        if y_data[i] not in seperated_dataset:
            seperated_dataset[y_data[i]]=[]
        (seperated_dataset[y_data[i]]).append(dataset[i])
    print len(seperated_dataset[0])    

    y_new=[]
    nx=[]
    ny=[]
    for i in seperated_dataset:
        threshold = 1.5*np.std(seperated_dataset[i]) 
        median = np.median(seperated_dataset[i])
        median_absolute_deviation_y = np.median([np.abs(y - median) for y in seperated_dataset[i]])
        modified_z_scores = [0.6745 * (y - median) / median_absolute_deviation_y for y in seperated_dataset[i]]
        x1=[]
        x2=[]
        for j in modified_z_scores:
            x1.append(j[0])
            x2.append(j[1])
            
        print "class ",i," maximum x1 ",max(x1)
        print "class ",i," maximum x2 ",max(x2)
        for j in range(len(seperated_dataset[i])):
            if np.abs(modified_z_scores)[j][0] <=1.1 and np.abs(modified_z_scores)[j][1] <=2 :
                nx.append(seperated_dataset[i][j][0])
                ny.append(seperated_dataset[i][j][1])
                y_new.append(i)
        print "length of nx after class ",i," is ",len(nx)        
        
        
   
    plt.scatter(nx,ny , c=y_new, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    #plt.savefig(file2)
    plt.show()
    


if args.data=="data_1.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_1.h5','/home/garvita/Desktop/Assignment2/data_outlier_1.png')
elif  args.data=="data_2.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_2.h5','/home/garvita/Desktop/Assignment2/data_outlier_2.png')
elif args.data=="data_3.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_3.h5','/home/garvita/Desktop/Assignment2/data_outlier_3.png')
elif args.data=="data_4.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_4.h5','/home/garvita/Desktop/Assignment2/data_outlier_4.png')
elif args.data=="data_5.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_5.h5','/home/garvita/Desktop/Assignment2/data_outlier_5.png')    
else:
        raise Exception("Invalid file name")
    
