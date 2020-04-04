'''
Classification for Stat287 Project 02
UNTOUCHABLE_MIXER
Script to go through KNN and Linear SVC classification
using data created in Project01 for training
and new data for testing. 
'''

## Imports ##
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Import utils to read in data
import utils


#Get training data from project 1
#Pull from utils python file
x_train, y_train = utils.project1_data()

###########################################
##Start with KNN
###########################################
# empty list that will hold cv scores
scores_unif = dict()
scores_dist = dict()

# perform 10-fold cross validation
for k in neighbors:
    for w in ['uniform','distance']:
        knn_brain = Pipeline([
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", KNeighborsClassifier(n_neighbors = k, 
                                             weights = w)),
                ])
        scores = cross_val_score(knn_brain, x_train, y_train, 
                             cv=10)
        if w == 'uniform':
            scores_unif[k] = [scores, scores.mean(), scores.std()]

        if w == 'distance':
            scores_dist[k] = [scores, scores.mean(), scores.std()]

#plot the different scores for the different parameters
def plot_missclass(scores, weight):
# changing to misclassification error
    missclass_err = [1 - m for s, m, sd in scores.values()]

    # determining best k
    optimal_k = neighbors[missclass_err.index(min(missclass_err))]
    print("The optimal number of neighbors is {}".format(optimal_k))
    print('With minimum mean misclassification error of {}'.format(min(missclass_err)))

    # plot misclassification error vs k
    plt.plot(neighbors, missclass_err, label = weight)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Mean Misclassification Error")
    plt.title('KNN')
    #plt.show()

plot_missclass(scores_unif, weight = 'Uniform')
plot_missclass(scores_dist, weight = 'Distance')
plt.legend(title = 'Weighting method')

#Distance weighting with 9 neighbors has the lowest misclassification error

## Now predict on new testing data with optimized model
#Get testing data
x_test, y_test = new_data()

knn_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", KNeighborsClassifier(n_neighbors = 9, 
                                             weights = 'distance')),
                                     ])
        
#Train with our chosen parameters
knn_fit = knn_brain.fit(x_train, y_train)
pred = knn_brain.predict(x_test)

# evaluate accuracy
print("accuracy: {}".format(accuracy_score(y_test, pred)))

metrics.confusion_matrix(y_test, pred)
metrics.plot_confusion_matrix(knn_fit, x_test, y_test)


###################################
## SVC
###################################

reg_params = np.linspace(1,10,20)
class_params = ['ovr', 'crammer_singer']

class_params = ['ovr']
ovr_scores = dict()
crammer_scores = dict()

for r in reg_params:
    for c in class_params:
        
        svm_brain = Pipeline([
                    ("vect", CountVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("clf", LinearSVC(C = r,
                                      multi_class = c,
                                      max_iter = 20000)),
                    ])
        scores = cross_val_score(svm_brain, x_train, y_train,
                             cv = 10)
        
        if c == 'ovr':
            #ovr_scores.append(scores.mean())
            ovr_scores[round(r, 2)] = [scores, scores.mean(), scores.std()]
        if c == 'crammer_singer':
            #crammer_scores.append(scores.mean())
            crammer_scores[round(r,2)] = [scores, scores.mean(), scores.std()]
            

missclass_err = [1 - m for s, m, sd in ovr_scores.values()]

    # determining best k
optimal_r = reg_params[missclass_err.index(min(missclass_err))]
print("The optimal regularization parameter is {}".format(optimal_r))
print('With minimum mean missclassification error of {}'.format(min(missclass_err)))

    # plot misclassification error vs k
plt.plot(reg_params, missclass_err)
plt.xlabel("Regularization paraemter")
plt.ylabel("Mean Misclassification Error")
#plt.title('KNN using {} weighting'.format(weight))
plt.show()


#Predict on New Test Data
svm_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC(C = 2.42, multi_class = 'ovr',
                          max_iter = 20000)),
                                     ])
        
#Train with our chosen parameters
svm_fit = svm_brain.fit(x_train, y_train)
pred = svm_brain.predict(x_test)

# evaluate accuracy
print("accuracy: {}".format(accuracy_score(y_test, pred)))

metrics.confusion_matrix(y_test, pred)
metrics.plot_confusion_matrix(svm_fit, x_test, y_test)

################
#################

#We can also compare crossvalidation accuracy scores of the two methods
#Using boxplot and the optimal parameters for both

data = [scores_dist[9][0],
        ovr_scores[2.42][0]]
plt.boxplot(data)
plt.xticks([1,2], ['KNN', 'Linear SVC'])
plt.xlabel('Accuracy')


################
#################
