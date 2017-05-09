#Copyright 2016 Qatar University and Carnegie Mellon University Qatar.

#For licensing information, contact the copyright holders.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
#OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
#AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Baljit Singh

#!/usr/bin/python

# The following code basically plots the Dirty-class data in a PCA
# reduced space. This code was used to generate a plot to represent the 
# the data graphically for research paper purposes


from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from numpy import genfromtxt
import numpy as np
from sklearn import svm
import csv
import csv_io2
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# command line input 
if len(sys.argv) > 3 or len(sys.argv) < 2:
	print "Usage: %s <data> [gamma]" % (sys.argv[0])
	sys.exit(-1)

if len(sys.argv) == 4:
	the_gamma = float(sys.argv[3])
else:
	the_gamma = 0.001


# Load the training data
training_raw = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
X = training_raw[:,1:] #training data
y = training_raw[:,:1] #training truth
y = y.reshape(-1)

# Linear classifier
classifier = svm.SVC(kernel="linear", class_weight="balanced")
classifier.fit(X, y)
pred = classifier.predict(X)
tp = 0
fp = 0
tn = 0
fn = 0
for i,j in enumerate(pred):
	if j == 0 and y[i] == 0:
		tn = tn + 1
	elif j == 0 and y[i] == 1:
		fn = fn + 1
	elif j == 1 and y[i] == 1:
		tp = tp + 1
	elif j == 1 and y[i] == 0:
		fp = fp + 1
print "tp: %d"%(tp)
print "tn: %d"%(tn)
print "fp: %d"%(fp)
print "fn: %d"%(fn)

# Reduce the feature space to 2D
reduced_data = PCA(n_components=3).fit_transform(X)

#separating the components
reduced_x = reduced_data[:,0]
reduced_y = reduced_data[:,1]
reduced_z = reduced_data[:,2]

# getting the components of clean data
clean_x = [reduced_x[i] for i,x in enumerate(y) if x == 0]
clean_y = [reduced_y[i] for i,x in enumerate(y) if x == 0]
clean_z = [reduced_z[i] for i,x in enumerate(y) if x == 0]

# getting the components of real rootkit data
real_x = [reduced_x[i] for i,x in enumerate(y) if x == 1]
real_y = [reduced_y[i] for i,x in enumerate(y) if x == 1]
real_z = [reduced_z[i] for i,x in enumerate(y) if x == 1]

# getting the components of wynthetic rootkit data
synth_x = [reduced_x[i] for i,x in enumerate(y) if x == 2]
synth_y = [reduced_y[i] for i,x in enumerate(y) if x == 2]
synth_z = [reduced_z[i] for i,x in enumerate(y) if x == 2]


print len(X)
print len(reduced_data)

# create a PCA-plot based on real rookit data, synthetic
# rootkit data and clean data
fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(clean_x, clean_y, clean_z, color='b', marker='o', label="Clean")
ax.scatter(real_x, real_y, real_z, color='g', marker='o', label="Real")
ax.scatter(synth_x, synth_y, synth_z, color='r', marker='o', label="Synthetic")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_title('Real and Synthetic Rootkit Traces in a PCA Reduced Space')

plt.legend(loc="lower right")

plt.show()
