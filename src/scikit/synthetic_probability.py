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

# The following code uses different classifiers to compare the most number of correctly classified
# rootkit techniques(labels). This is manually done by comparing the the labels from the prediction
# array



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
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

# command line input 
if len(sys.argv) > 4 or len(sys.argv) < 3:
	print "Usage: %s <training-clean> <testing-dirty> [gamma]" % (sys.argv[0])
	sys.exit(-1)

if len(sys.argv) == 4:
	the_gamma = float(sys.argv[3])
else:
	the_gamma = 0.00000001


# Load the mixed training data
training_raw = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
training_data = training_raw[:,1:]
training_truth = training_raw[:,:1]
training_truth = training_truth.reshape(-1)

# Load the testing data
testing_raw = np.loadtxt(sys.argv[2], delimiter=",", skiprows=1)
testing_data = testing_raw[:,1:]
testing_truth = testing_raw[:,:1]

X_train = training_data 
y_train = training_truth
X_test = testing_data
y_test = testing_truth

# array of different classifiers objects which
# will be used for prediction purposes
classifiers = []
classifiers.append( (tree.DecisionTreeClassifier(class_weight="balanced", max_features='auto'), "Decision Tree", 0) )
classifiers.append( (svm.SVC(kernel="rbf", probability=True, gamma=0.000001, class_weight="balanced"), "SVM (RBF)", 0) )
classifiers.append( (svm.SVC(kernel="linear", probability=True, class_weight="balanced"), "SVM (Linear)", 0) )
classifiers.append((OneVsRestClassifier(svm.SVC(kernel="linear", class_weight="balanced")), "One vs Rest", 0) )

oneCounter = 0
twoCounter = 0
threeCounter = 0
cleanCounter = 0
fiveCounter = 0
sixCounter = 0
sevenCounter = 0
eightCounter = 0

# loop through the classifier array and check which
# one works the best
for (classifier, name, clftype) in classifiers:
	classifier.fit(X_train, y_train)
	pred = classifier.predict(X_test)
	
	print "Classifier Used:"+name
	print "----------------------"
	for probs_element in pred:
		if probs_element == 1.0:
			oneCounter = oneCounter + 1
		elif probs_element == 2.0:
			twoCounter = twoCounter + 1
		elif probs_element == 3.0:
			threeCounter = threeCounter + 1
		elif probs_element == 0.0:
			cleanCounter = cleanCounter + 1
		elif probs_element == 5.0:
			fiveCounter = fiveCounter + 1
		elif probs_element == 6.0:
			sixCounter = sixCounter + 1
		elif probs_element == 7.0:
			sevenCounter = sevenCounter + 1
		elif probs_element == 8.0:
			eightCounter = eightCounter + 1

	# print out the counters for each synthetic rootkit
	# technique combination for probaility purposes
	print "Clean Data:",cleanCounter
	print "DKOM:",oneCounter
	print "IRP:",twoCounter
	print "SSDT:",threeCounter
	print "File IRP Process SSDT:",fiveCounter
	print "Netfilter IRP Filefilter SSDT:",sixCounter
	print "Netfilter IRP Process SSDT:",sevenCounter
	print "Filefilter IRP SSDT:",eightCounter
	print "------------------------------\n"
	