import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
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
    
    x=[]
    y=[]
    for i in dataset:
        x.append(i[0])
        y.append(i[1])
    #print x
    #print y
    count={}
    y_data=f['y'][:]
    for i in y_data:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    print(count)
    plt.scatter(x,y , c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    #plt.savefig(file2)
    plt.show()
    


if args.data=="data_1.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_1.h5','/home/garvita/Desktop/Assignment2/data_1.png')
elif  args.data=="data_2.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_2.h5','/home/garvita/Desktop/Assignment2/data_2.png')
elif args.data=="data_3.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_3.h5','/home/garvita/Desktop/Assignment2/data_3.png')
elif args.data=="data_4.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_4.h5','/home/garvita/Desktop/Assignment2/data_4.png')
elif args.data=="data_5.h5":
    Part_A('/home/garvita/Desktop/Assignment2/data_5.h5','/home/garvita/Desktop/Assignment2/data_5.png')    
else:
        raise Exception("Invalid file name")
    
