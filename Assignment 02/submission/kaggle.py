import json
import numpy as np
from sklearn.utils import resample
from sklearn import svm
import re
import csv
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

indices=set()
max_indices=0
labels=[]
features=[]

#load json file and obtain labels and features
with open('/home/garvita/Desktop/Assignment2/train.json') as json_file:  
    data = json.load(json_file)
    for p in data:
        labels.append(p['Y'])
        features.append(p['X'])



    


#print len(features)

#remove dataset having label 0
for i in range(len(labels)):
  if(labels[i])==0:
    del features[i]
    

#print len(features)

for i in labels:
  if i==0:
    labels.remove(i)


  
#print len(labels)    

#count of number of samples in each class        

label_c={}

for i in labels:
    if i in label_c:
        label_c[i]+=1
    else:
        label_c[i]=1
        
#print label_c

#avg number of samples
summ=0
for i in label_c:
    summ+=label_c[i]     
summ=summ/5
#count number of times features occur and take only those indices which occur
features_count={}
for i in features:
    for j in range(len(i)):
        indices.add(i[j])
        if i[j] not in features_count:
            features_count[i[j]]=1
        else:
            features_count[i[j]]+=1
    
#print features_count
#print len(features)
            
#removing irrelevant and unused indices            
updated=[]
for i in features_count:
    if(features_count[i]>=10):
        updated.append(i)
print len(updated)        
features_matrix=np.zeros(shape=(len(labels),len(updated)))


#create feature matrix
#with open('/home/garvita/Desktop/Assignment2/featurematrix.txt','w') as f:
for i in range(len(features)):   #each row
  for j in range(len(features[i])): #each column in a row
    if features[i][j] in updated:
      features_matrix[i,updated.index(features[i][j])] = np.array([1])
        #for k in range(len(updated)):
            #f.write('%s' %features_matrix[i][k])
            #f.write(" ")
        #f.write("\n")'''
print "length of updated ",len(updated)
print features_matrix

clf=svm.LinearSVC()
clf.fit(features_matrix,labels)

test=[]
with open('/home/garvita/Desktop/Assignment2/test.json') as json_file:  
    data = json.load(json_file)
    for p in data:
        test.append(p['X'])

test_matrix=np.zeros(shape=(len(test),len(updated)))
for i in range(len(test)):
  for j in range(len(test[i])):
    if test[i][j] in updated:
      test_matrix[i,updated.index(test[i][j])] = np.array([1])
      
prediction=clf.predict(test_matrix)  

data={}
j=1
for i in prediction:
  data[j]=i
  j=j+1

f = open('/home/garvita/Desktop/Assignment2/output.csv', "wt") 
#"w" indicates that you're writing strings to the file

try:
    writer = csv.writer(f)
    writer.writerow( ('Id', 'Expected') )
    for key in data.keys():
        writer.writerow( (key,data[key]) )

finally:
    f.close() 
  

  


    
        
              
                


    

