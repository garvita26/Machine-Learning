import os
import os.path
import argparse
import h5py
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


if args.test_data == 'Data/part_A_train.h5':
        X,Y=load_h5py(args.test_data)
        #path pf the saved model with filename
        with open(args.weights_path,"rb") as f:
                var=pickle.load(f)
        #print var        
        prediction =var.predict(X)
        with open(args.output_preds_file,"wb") as fi:
                for i in prediction:
                        fi.write(str(i))
                        fi.write("\n")
        fi.close()	
elif args.test_data == 'Data/part_B_train.h5':
        X,Y=load_h5py(args.test_data)
        with open(args.weights_path,"rb") as f:
                var=pickle.load(f)
        prediction =var.predict(X)
        with open(args.output_preds_file,"wb") as fi:
                for i in prediction:
                        fi.write(str(i))
                        fi.write("\n")
        fi.close()	
	
elif args.test_data == 'Data/part_C_train.h5':
        X,Y=load_h5py(args.test_data)
        with open(args.weights_path,"rb") as f:
                var=pickle.load(f)
        prediction =var.predict(X)
        with open(args.output_preds_file,"wb") as fi:
                for i in prediction:
                        fi.write(str(i))
                        fi.write("\n")
        fi.close()	
        
	
else:
	raise Exception("Invald test file")
