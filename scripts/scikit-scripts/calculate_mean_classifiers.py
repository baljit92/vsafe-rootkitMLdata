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

# The following code calculates te senstivity, specificity, fall out rate and false negative rate
# for different classifiers such as Naives Bayes, SVM(with various kernels) and Decision Trees.
# These calculates help us decide which classifier works best to correctly detect the malicious 
# activity. 



from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
from scipy.sparse import hstack
from numpy import genfromtxt
import numpy as np
from sklearn import svm
import csv
import csv_io2
import sys
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from scipy import interp

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

from sklearn import tree

# command line input 
if len(sys.argv) > 5 or len(sys.argv) < 4:
	print "Usage: %s <training-clean> <training-dirty> <testingdata> [gamma]" % (sys.argv[0])
	sys.exit(-1)

if len(sys.argv) == 5:
	the_gamma = float(sys.argv[4])
else:
	the_gamma = 0.00000001


# Load the clean training data
trainingc_raw = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
trainingc_data = trainingc_raw[:,1:]
trainingc_truth = trainingc_raw[:,:1]
trainingc_truth = trainingc_truth.reshape(-1)

# Load the dirty training data
trainingd_raw = np.loadtxt(sys.argv[2], delimiter=",", skiprows=1)
trainingd_data = trainingd_raw[:,1:]
trainingd_truth = trainingd_raw[:,:1]
trainingd_truth = trainingd_truth.reshape(-1)

# Create the training data by combining
# clean and dirty
training_data = np.append(trainingc_data, trainingd_data, axis=0)
training_truth = np.append(trainingc_truth, trainingd_truth)

# Load the testing data
testing_raw = np.loadtxt(sys.argv[3], delimiter=",", skiprows=1)
testing_data = testing_raw[:,1:]
testing_truth = testing_raw[:,:1]

X_train = training_data 
y_train = training_truth
X_test = testing_data
y_test = testing_truth

# classifier objects
classifiers = []
classifiers.append( (tree.DecisionTreeClassifier(class_weight="balanced", max_features='auto'), "Decision Tree", 0) )
classifiers.append( (GaussianNB(), "Naive Bayes", 0) )
classifiers.append( (svm.OneClassSVM(kernel="linear"), "OC SVM (Linear)", 1) )
classifiers.append( (svm.SVC(kernel="rbf", probability=True, gamma=the_gamma, class_weight="balanced"), "SVM (RBF)", 0) )
classifiers.append( (svm.SVC(kernel="linear", probability=True, class_weight="balanced"), "SVM (Linear)", 0) )


plt.figure()

trVal = range(0,50)
mean_tprVal = 0.0
mean_fprVal = 0.0
mean_tnrVal = 0.0
mean_fnrVal = 0.0

for (classifier, name, clftype) in classifiers:
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for b in trVal:
		if clftype == 1: # one class classifier
			fitted = classifier.fit(trainingc_data)
		else: # multi-class classifier
			fitted = classifier.fit(X_train, y_train)
		
		# Get the predictions 
		pred = classifier.predict(X_test)

		for i,j in enumerate(pred):
			# Convert predictions to 1,0 for unsupervised models...
			if clftype == 1:
				if j == 1:
					j = 0;
				elif j == -1:
					j = 1
				else:
					print "panic"

			if j == 0 and y_test[i] == 0:
				tn = tn + 1
			elif j == 0 and y_test[i] == 1:
				fn = fn + 1
			elif j == 1 and y_test[i] == 1:
				tp = tp + 1
			elif j == 1 and y_test[i] == 0:
				fp = fp + 1
	tpr = float(tp)/(tp+fn)
	fpr = float(fp)/(fp+tn)
	tnr = float(tn)/(tn+fp)
	fnr = float(fn)/(fn+tp)

	print "\nData for %s"%name
	print "True Positive: %f"%tpr
	print "True Negative: %f"%tnr
	print "False Positive: %f"%fpr
	print "False Negative: %f"%fnr


	# Plot the decision tree results
	if name == "Decision Tree":
		tree.export_graphviz(classifier, out_file='tree.dot')

	# Plot the ROC
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	tr = range(0,50)
	for b in tr:
		if clftype == 1:			
			fitted = classifier.fit(trainingc_data)		
		else:
			fitted = classifier.fit(X_train, y_train)

		try:
			y_score = fitted.decision_function(X_test)
		except:
			hmm = fitted.predict_proba(X_test)
			y_score = hmm[:, 1]

		# Because we train the unsupervised models on clean, the convention is swapped
		if clftype == 1:
			pos_label = 0
		else:
			pos_label = 1
		roc_fpr, roc_tpr, roc_thresh = roc_curve(y_test, y_score, pos_label=pos_label)
		mean_tpr += interp(mean_fpr, roc_fpr, roc_tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(roc_fpr, roc_tpr)
	mean_tpr /= len(tr)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)

	plt.plot(mean_fpr, mean_tpr, label='%s' % (name))

# Plot an ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Accuracy of Classifiers')
plt.legend(loc="lower right")
plt.show()
