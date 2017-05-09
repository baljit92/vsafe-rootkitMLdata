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

# The following code uses Neural Networks model that can take in decided features and attempt 
# to predict malicious or benign labels each piece of data based on the HPCs. 
# The data is passed through variable unit sizes, learning rates and epochs to decide what 
# parameters produce more accurate results.

from __future__ import division
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.sparse import hstack
#from matplotlib import pyplot
from numpy import genfromtxt
import numpy as np
from sklearn import svm
from sklearn import datasets, metrics
import csv
import csv_io2
import sys
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging



logging.basicConfig()

# command line input 
if len(sys.argv) > 3 or len(sys.argv) < 2:
	print "Usage: %s <data> [gamma]" % (sys.argv[0])
	sys.exit(-1)

# check if gamma is speicifed by the user 
if len(sys.argv) == 4:
	the_gamma = float(sys.argv[3])
else:
	the_gamma = 0.001


# Load the raw training data
training_raw = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
X = training_raw[:,1:]
y = training_raw[:,:1]
y = y.reshape(-1)

# Load the raw testing data
testing_raw = np.loadtxt(sys.argv[2], delimiter=",", skiprows=1)
testing_data = testing_raw[:,1:]
testing_truth = testing_raw[:,:1]

# Data division into labels and fetures for training and 
# testing data
X_train = X
y_train = y
X_test = testing_data
y_test = testing_truth

learning_rate = [0.01, 0.001, 0.0001, 0.00001]

# Testing out the neural networks with a range of 
# learning rates, different number of units, different 
# epochs  

for index in range(0,5):
	for unit_size in range(5,101):
		for epoch in range(50,101):
			print "------------------------------------------------------------------"
			print "\tHidden Unit Size : %d, Epoch: %d, Learning Rate: %0.2f" % (unit_size, epoch, learning_rate[index])
			print "------------------------------------------------------------------"
			classifier = Classifier(
			    layers=[
			        Layer("Rectifier", units=15),
			        Layer("Rectifier", units=unit_size),
			        Layer("Softmax", units = 2)],
			    learning_rate=learning_rate[index],
			    n_iter=epoch)
			classifier.fit(X_train, y_train)

			pred = classifier.predict(X_test)
			pred_train = classifier.predict(X_train)
			score = metrics.accuracy_score(y_test, pred)
			
			# calculating the number of true positive, true 
			# negatives, false positives, false negatives
			# for training data
			tp = 0
			fp = 0
			tn = 0
			fn = 0
			accuracy = 0
			for i,j in enumerate(pred_train):
				if j == 0 and y_train[i] == 0:
					tn = tn + 1
				elif j == 0 and y_train[i] == 1:
					fn = fn + 1
				elif j == 1 and y_train[i] == 1:
					tp = tp + 1
				elif j == 1 and y_train[i] == 0:
					fp = fp + 1

			num = (tp + tn)
			denom = (tp + fp + tn + fn)
			score = num/denom
			
			print "\n"
			
			print "------------- Training ----------------"
			print "Score:%0.2f"%(score)
			print "True Positive:%d, True Negative:%d " %(tp, tn)
			print "False Positive:%d, False Negative:%d" %(fp, fn)


			# calculating the number of true positive, true 
			# negatives, false positives, false negatives
			# for testing data
			tp = 0
			fp = 0
			tn = 0
			fn = 0
			accuracy = 0
			for i,j in enumerate(pred):
				if j == 0 and y_test[i] == 0:
					tn = tn + 1
				elif j == 0 and y_test[i] == 1:
					fn = fn + 1
				elif j == 1 and y_test[i] == 1:
					tp = tp + 1
				elif j == 1 and y_test[i] == 0:
					fp = fp + 1

			num = (tp + tn)
			denom = (tp + fp + tn + fn)
			score = num/denom
			
			print "\n"
			print "------------- Testing ----------------"
			print "Score:%0.2f"%(score)
			print "True Positive:%d, True Negative:%d, Numerator:%d" %(tp, tn, num)
			print "False Positive:%d, False Negative:%d, Denominator:%d" %(fp, fn, denom)
			print "\n"
			print "\n"

			

