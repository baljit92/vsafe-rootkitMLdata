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


# The following piece of code uses TensorFlow to train for a certain number of epochs,
# differenet splits of training data and evaluate a feed-forward neural network. We 
# calculate the mean accuracy over the labels

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.sparse import hstack
import tensorflow as tf
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

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# this network is the same as the previous 
# one except with an extra hidden layer + dropout
def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden): 

    X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.matmul(h2, w_o)

# Used to convert the sparse data to tensorflow
# one hot representation
def convertOneHot(data):
        y=np.array([int(i[0]) for i in data])
        y_onehot=[0]*len(y)
        for i,j in enumerate(y):
                y_onehot[i]=[0]*(y.max() + 1)
                y_onehot[i][j]=1
        return (y,y_onehot)



if len(sys.argv) > 4 or len(sys.argv) < 3:
        print "Usage: %s <training-clean> <testing-dirty> [gamma]" % (sys.argv[0])
        sys.exit(-1)

if len(sys.argv) == 4:
        the_gamma = float(sys.argv[3])
else:
        the_gamma = 0.00000001

# Load the training data
training_raw = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
training_data = training_raw[:,1:]
training_truth, training_truth_onehot = convertOneHot(training_raw)

# Load the testing data
testing_raw = np.loadtxt(sys.argv[2], delimiter=",", skiprows=1)
testing_data = testing_raw[:,1:]
testing_truth, testing_truth_onehot = convertOneHot(testing_raw)

A=training_raw.shape[1]-1 # Number of features, Note first is y
B=len(training_truth_onehot[0])


X = tf.placeholder("float", [None, A])
Y = tf.placeholder("float", [None, B])

# initialize input, hidden and output layers
# for the graph
w_h = init_weights([A, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, B])


p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute cost
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# train for 100 epochs
for i in range(100):
    for start, end in zip(range(0, len(training_data), 128), range(128, len(training_data), 128)):
        sess.run(train_op, feed_dict={X: training_data[start:end], Y: training_truth_onehot[start:end],
                                      p_keep_input: 0.8, p_keep_hidden: 0.5})
    print i, np.mean(np.argmax(testing_truth_onehot, axis=1) ==
                     sess.run(predict_op, feed_dict={X: testing_data, Y: testing_truth_onehot,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))