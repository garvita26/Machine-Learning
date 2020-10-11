import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import h5py



f=h5py.File('/home/garvita/Desktop/Assignment2/data_1.h5', "r+")
#print f
#print "keys: %s" %f.keys()
dataset=f['x'][:]
    
x1=[]
x2=[]
for i in dataset:
    x1.append(i[0])
    x2.append(i[1])


y_data=f['y'][:]    
# figure number



# fit the model

clf = svm.SVC(kernel='rbf', gamma=2)
clf.fit(dataset, y_data)

# plot the line, the points, and the nearest vectors to the plane
#plt.figure(fignum, figsize=(4, 3))
plt.clf()

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x1, x2, c=y_data, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

plt.axis('tight')
x_min = -4
x_max = 4
y_min = -4
y_max = 4

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
#plt.figure(fignum, figsize=(4, 3))
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
   
plt.show()
