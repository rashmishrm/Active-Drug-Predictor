import re
import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC,NuSVC
from sklearn import cross_validation, linear_model
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.linear_model import Perceptron

from sklearn import svm

train_file = "train.dat"
test_file = "test.dat"

feature_size = 100001
k_folds = 10




def load(filename, ftype):
    with open(filename, "r") as rfile:
        lines = rfile.readlines()

    if ftype == "train":
        labels = [int(l[0]) for l in lines]
        for index, item in enumerate(labels):
            if (item == 0):
                labels[index] = -1
        docs = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]

    else:
        labels = []
        docs = [re.sub(r'[^\w]', ' ',l).split() for l in lines]

    features = []

    for doc in docs:
        line = [0]*feature_size
        for index, val in enumerate(doc):
            line[int(val)] = 1
        features.append(line)

    return features, labels


print 'Starting processing for drug activity prediction'

print "Loading training data"
# Loading train.dat file
features, labels = load(train_file, "train")

#Using Dimensionality Reduction on train data
print "Reducing Dimensions using Truncated SVD on train data"


svd_trunc = TruncatedSVD(algorithm='randomized', n_components=1500, n_iter=50, random_state=42)

svd_trunc_m = svd_trunc.fit(features, labels)
reduced_features = svd_trunc_m.transform(features)



#Using oversampling SMOTE

print "Oversampling data using SMOTE!"
sm = SMOTE(random_state=42,kind='svm')
reduced_features, labels = sm.fit_sample(reduced_features, labels)


#processing test data
print "Loading test data"


test_features, test_labels = load(test_file, "test")


print "Reducing Dimensions using Truncated SVD on test data"
test_reduced_features = svd_trunc_m.transform(test_features)

# Classifying
names = ["Decision Tree"]
classifiers = [DecisionTreeClassifier(random_state=53,class_weight={-1: 1, 1: 1.5})]


print 'Starting classification!!'

for name, clf in zip(names, classifiers):
    print 'Report on ' + name
    cv_predicted = cross_val_predict(clf, reduced_features, labels, cv=k_folds)

    print metrics.classification_report(labels, cv_predicted)

    scores = cross_validation.cross_val_score(clf, reduced_features, labels)

    print '\nCross validation scores: '
    print scores.mean()

    #training classifier
    clf.fit(reduced_features, labels)

    # Predict test labels
    test_predicted = clf.predict(test_reduced_features)

    print 'Test predicted for ' + name

    result_file = 'format.dat'

    print 'Output stored in', result_file

    output = open(result_file, 'w')
    for t in test_predicted:
        if int(t) == -1:
            t = 0
        output.write(str(t))
        output.write("\n")
    output.close()

print 'Finished!'
