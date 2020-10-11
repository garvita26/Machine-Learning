import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.manifold import TSNE
from sklearn import svm
import scipy


def linearkernel(x,y):
    C=1
    return np.dot(x,y.T)

def gaussiankernel(x,y):
    #print x,y
    gamma=2
    #print np.shape(np.exp(-gamma*scipy.spatial.distance.cdist(x,y)))
    return np.exp(-gamma*scipy.spatial.distance.cdist(x,y))

f=h5py.File('/home/garvita/Desktop/Assignment2/data_5.h5', "r+")
#print f
#print "keys: %s" %f.keys()
dataset=f['x'][:]
    
x1=[]
x2=[]
for i in dataset:
    x1.append(i[0])
    x2.append(i[1])

y_data=f['y'][:]
mu=np.mean
clf = svm.SVC(kernel=gaussiankernel)
clf.fit(dataset, y_data)
plt.clf()


plt.scatter(x1, x2, c=y_data, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
#plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')
h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = min(x1) - 1, max(x1) + 1
y_min, y_max = min(x2) - 1, max(x2) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print np.shape(Z)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
print "after ",np.shape(Z)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
'''plt.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.Paired)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())'''
   
plt.show()

