''' Script for STAT 287 Project 2.
Code for both paper 1 and paper 2
First half trains and tests the following 5 different classification methods
using project 01 database:
    KNN, LinearSVC, MultinomialNB, Random Forests, and Ridge Classifier
These results are written up in paper 1.

Second half uses the best classifier from part 1 (Ridge Classifier)
and explores how accuracy improves when training using external data
provided for project 02. These results are written up in paper 2
'''

##############
#Imports
##############
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
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier


import numpy as np
import time
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Import utils to read in data
import utils


#Get data from project 1
x_train, y_train = utils.project1_data()
#Get newly created testing data
x_test, y_test = utils.new_data()

#List of models used in this analysis
models = ['KNN', 'LinearSVC', 'MultinomialNB',
          'Random Forest', 'Ridge Classifier']


###############
## Functions ##
###############

def check_redundancy(data_large, data_small):
    '''Function to check for data redundancy.
    Returns a list of docs that are redundant 
    and prints the number of redundant docs.
    data_large: list 1 of data to check
    data_small: list 2 of data to check, typicaly smaller list 1
    '''
    redundant = list()
    for d in data_small:
        if d in data_large:
            redundant.append(d)
    print(len(redundant))
    return(redundant)

def cv_func(pipe, tuned_parameters, x_train, y_train ):
    ''' Function to run GridSearchCV and fit brain.
    Outputs a trained brain
    Pipe: pipeline to use for classifing
    tuned_parameters: dictionary of parameters for GridSearchCV
    x_train: x (docs) training data
    y_train: y (targets) training data
    '''
    
    clf = GridSearchCV(pipe, tuned_parameters, cv = 10)
    clf.fit(x_train, y_train)
    return(clf)

def cv_plot(cv_results, param1, param2, legend_title = None,
            xlabel = None, main_title = None):
    ''' Function to plot cv results. 
    Output a plot of cv accuracy
    cv_results: cv_func.cv_results_
    param1: list of primary parameters
    param2: list of secondary parameters
    legend_title: Include if a legend for secondary parameters is desired
    xlabel: include if xlabel is needed
    main_title: include if main title is desired
    '''
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(param1), len(param2))
    for idx, val in enumerate(param2):
        plt.plot(param1, scores_mean[:,idx], label = val)
    
    
    #plt.plot()
    if legend_title != None:
        plt.legend(title = legend_title)
    plt.ylabel('Mean Accuracy')
    plt.xlabel(xlabel)
    plt.title(main_title)
    #plt.show()
    
        
def final_metrics(brain, label, x_train, y_train, x_test, y_test):
    ''' Function to calculate final metrics for testing deployment. 
    Ouput the prediction targets and a list consisting of a specified label,
        accuracy score, training time, and testing time
    brain: pipeline brain 
    label: any desired label
    x_train: x (docs) training data
    y_train: y (targets) training data
    x_test: x testing data
    y_test: y testing data
    '''
    
    start_time = time.time()
    fit = brain.fit(x_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    pred = brain.predict(x_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(y_test, pred)
    print("accuracy: {}".format(accuracy))

    model = np.array([label, accuracy, train_time, test_time])
    metrics.plot_confusion_matrix(fit, x_test, y_test)
    return(pred, model.reshape(1,4))
    
check_redundancy(x_train, x_test)

    
################################
##          KNN               ##
################################
    
knn_brain = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", KNeighborsClassifier()),
        ])
    
#List of range of number of neighbors to try
neighbors = list(range(1, 50, 2))
weights = ['uniform', 'distance']
knn_parameters = {'clf__n_neighbors': neighbors,
                     'clf__weights' : weights}
    
knn_cv = cv_func(pipe = knn_brain, tuned_parameters = knn_parameters,
                 x_train = x_train, y_train = y_train)
cv_plot(knn_cv.cv_results_,
        neighbors, weights, legend_title = 'Weighting Method',
        xlabel = 'Numer of Neighbors, K', 
        main_title = 'KNN')
plt.plot([knn_cv.best_params_['clf__n_neighbors'],]*2,
         [0.2,knn_cv.best_score_],
         linestyle = '--', color = 'black',
         ms = 8)
plt.show()


## Now train/test with best parameters
best_params = knn_cv.best_params_

knn_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", KNeighborsClassifier(n_neighbors = best_params['clf__n_neighbors'], 
                                             weights = best_params['clf__weights'])),
                                     ])

knn_pred, knn = final_metrics(knn_brain, 0,
                              x_train, y_train, x_test, y_test)

knn_wrong = list()
for p in range(len(knn_pred)):
    if knn_pred[p] != y_test[p]:
        print(x_test[p], y_test[p], knn_pred[p])
        knn_wrong.append(p)

################################
##    LinearSVC               ##
################################

svc_brain = Pipeline([
                    ("vect", CountVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("clf", LinearSVC(max_iter = 20000))
                    ])
reg_params = np.linspace(1,10,20)
class_params = ['ovr']
svc_parameters = {'clf__C': reg_params,
                    'clf__multi_class': class_params}

svc_cv = cv_func(pipe = svc_brain, tuned_parameters = svc_parameters,
                 x_train = x_train, y_train = y_train)
cv_plot(svc_cv.cv_results_,
        reg_params, class_params,
        xlabel = 'Regularization parameter', 
        main_title = 'Linear SVC')
plt.plot([svc_cv.best_params_['clf__C'],]*2,
         [0.764,svc_cv.best_score_],
         linestyle = '--', color = 'black',
         ms = 8)
plt.show()


## Now train/test with best parameters
best_params = svc_cv.best_params_

svc_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC(C = best_params['clf__C'], 
                                             multi_class = best_params['clf__multi_class'])),
                                     ])

svc_pred, svc = final_metrics(svc_brain, 1,
                              x_train, y_train, x_test, y_test)

svc_wrong = list()
for p in range(len(svc_pred)):
    if svc_pred[p] != y_test[p]:
        print(x_test[p], y_test[p], svc_pred[p])
        svc_wrong.append(p)

################################
##    MultinomialNB           ##
################################

mnb_brain = Pipeline([
                    ("vect", CountVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("clf", MultinomialNB())
                    ])
smooth_params = np.linspace(0,1,25)
prior_params = [True, False]

mnb_parameters = {'clf__alpha': smooth_params,
                    'clf__fit_prior': prior_params}

mnb_cv = cv_func(pipe = mnb_brain, tuned_parameters = mnb_parameters,
                 x_train = x_train, y_train = y_train)
mean_scores = mnb_cv.cv_results_['mean_test_score']
mean_scores = np.array(mean_scores).reshape(len(smooth_params), 
                       len(prior_params))

best_1 = list(mean_scores[:,0]).index(max(mean_scores[:,0]))
best_2 = list(mean_scores[:,1]).index(max(mean_scores[:,1]))

cv_plot(mnb_cv.cv_results_,
        smooth_params, prior_params,
        xlabel = 'Smoothing Parameter', 
        main_title = 'Multinomial Naive Bayes')
plt.legend(('Class Prior', 'Uniform'), title = 'Prior Fit')
plt.plot([smooth_params[best_1], ] * 2,
         [0.62, max(mean_scores[:, 0])],
         linestyle = '--', color = 'blue')
plt.plot([smooth_params[best_2], ] * 2,
         [0.62, max(mean_scores[:, 1])],
         linestyle = '--', color = 'orange')
plt.show()


## Now train/test with best parameters
best_params = mnb_cv.best_params_

mnb_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB(alpha = best_params['clf__alpha'], 
                                             fit_prior = best_params['clf__fit_prior'])),
                                     ])

mnb_pred, mnb = final_metrics(mnb_brain, 2,
                              x_train, y_train, x_test, y_test)

mnb_wrong = list()
for p in range(len(mnb_pred)):
    if mnb_pred[p] != y_test[p]:
        print(x_test[p], y_test[p], mnb_pred[p])
        mnb_wrong.append(p)

################################
##    Random Forest           ##
################################

rf_brain = Pipeline([
                    ("vect", CountVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("clf", RandomForestClassifier(random_state = 10))
                    ])
trees_params = list(range(50, 300, 10))
maxft_params = [None, 'sqrt']

rf_parameters = {'clf__n_estimators': trees_params,
                    'clf__max_features': maxft_params}

rf_cv = cv_func(pipe = rf_brain, tuned_parameters = rf_parameters,
                 x_train = x_train, y_train = y_train)
mean_scores = rf_cv.cv_results_['mean_test_score']
mean_scores = np.array(mean_scores).reshape(len(trees_params), 
                       len(maxft_params))

best_1 = list(mean_scores[:,0]).index(max(mean_scores[:,0]))
best_2 = list(mean_scores[:,1]).index(max(mean_scores[:,1]))

cv_plot(rf_cv.cv_results_,
        trees_params, maxft_params,
        legend_title = 'Max Features',
        xlabel = 'Number of Trees', 
        main_title = 'Random Forest Classifier')
plt.legend(('n_features', 'sqrt(n_features)'), title = 'Max Features')
plt.plot([trees_params[best_1], ] * 2,
         [0.68, max(mean_scores[:, 0])],
         linestyle = '--', color = 'blue')
plt.plot([trees_params[best_2], ] * 2,
         [0.68, max(mean_scores[:, 1])],
         linestyle = '--', color = 'orange') 
plt.show()



## Now train/test with best parameters
best_params = rf_cv.best_params_

rf_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RandomForestClassifier(n_estimators = 190, 
                                             max_features = best_params['clf__max_features'],
                                             random_state = 10)),
                                     ])

rf_pred, rf = final_metrics(rf_brain, 3,
                              x_train, y_train, x_test, y_test)

rf_wrong = list()
for p in range(len(rf_pred)):
    if rf_pred[p] != y_test[p]:
        print(x_test[p], y_test[p], rf_pred[p])
        rf_wrong.append(p)

################################
##    Ridge Classifier        ##
################################

rc_brain = Pipeline([
                    ("vect", CountVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("clf", RidgeClassifier())
                    ])
reg_params = np.linspace(0.01,1,25)


rc_parameters = {'clf__alpha': reg_params}

rc_cv = cv_func(pipe = rc_brain, tuned_parameters = rc_parameters,
                 x_train = x_train, y_train = y_train)
mean_scores = rc_cv.cv_results_['mean_test_score']
mean_scores = np.array(mean_scores).reshape(len(reg_params), 
                       1)

best_1 = list(mean_scores[:,0]).index(max(mean_scores[:,0]))

cv_plot(rc_cv.cv_results_,
        reg_params, ['na'],
        xlabel = 'Regularization', 
        main_title = 'Ridge Classifier')
plt.plot([reg_params[best_1], ] * 2,
         [0.750, max(mean_scores[:, 0])],
         linestyle = '--', color = 'blue')
plt.show()

    

## Now train/test with best parameters
best_params = rc_cv.best_params_

rc_brain = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RidgeClassifier(alpha = best_params['clf__alpha'])),
                                     ])

rc_pred, rc = final_metrics(rc_brain, 4,
                              x_train, y_train, x_test, y_test)

rc_wrong = list()
for p in range(len(rc_pred)):
    if rc_pred[p] != y_test[p]:
        print(x_test[p], y_test[p], rc_pred[p])
        rc_wrong.append(p)

################################
##    Model Comparisons       ##
################################


all_models = np.concatenate((knn, svc, mnb, rf, rc), axis = 0)

all_models = np.array( sorted(all_models, key = lambda x:x[1]) )
y_pos = np.arange(len(models))

#Bar chart comparing accuracies
plt.barh(y_pos, all_models[:, 1], label = 'Accuracy')
plt.yticks(y_pos,
           [models[int(all_models[0,0])],
                   models[int(all_models[1,0])],
                   models[int(all_models[2,0])],
                   models[int(all_models[3,0])],
                   models[int(all_models[4,0])]])
plt.xlabel('Accuracy')
for i, v in enumerate(all_models[:, 1]):
    print(i)
    print(v)
    plt.text(v - 0.09 , i - 0.08, str(round(v,3)), color = 'white',
             fontweight = 'bold')

plt.show()

    

## Plot training and testing time using project 01 data
width = 0.4
plt.barh(y_pos, all_models[:, 3], width, label = 'Testing Time',
         color = 'orange')
plt.barh(y_pos + width, all_models[:, 2], width, label = 'Training Time',
         color = 'purple')
plt.yticks(y_pos,
           [models[int(all_models[0,0])],
                   models[int(all_models[1,0])],
                   models[int(all_models[2,0])],
                   models[int(all_models[3,0])],
                   models[int(all_models[4,0])]])
plt.legend()
plt.xlabel('Time (seconds)')
plt.show()


#Get all docs (by index) that are misclassified
all_wrong = knn_wrong + svc_wrong + mnb_wrong + rf_wrong + rc_wrong
all_wrong_c = dict(Counter(all_wrong))

#Get docs that are misclassified by all methods
idx_all = [i for i, c in all_wrong_c.items() if c == 5]
docs_all = [x_test[i] for i in idx_all]

#Docs that are misclassified by our best two classifiers
docs_rc = [x_test[i] for i in rc_wrong]
docs_svc = [x_test[i] for i in svc_wrong]

#Get a count of targets that are misclassified
targets = [y_test[i] for i in all_wrong]
targets_c = dict(Counter(targets))


################################################
## Compare with external data  (paper 2)      ##
################################################

#Read in cleaned external data
data_x, data_y = utils.preprocess_data(cleaned_version=True)

#Get length of cleaned external data
total_data = len(data_x)
num_train = int(total_data * 0.8)
num_test = total_data - num_train

#split external data into train/test split
#test split held off to test deployment
#test is 20% of total data
x_train_ext, x_test_ext, y_train_ext, y_test_ext = train_test_split(data_x, data_y, random_state=10, 
                 train_size=num_train, test_size=num_test)

#Get breakdown of targets in test data
test_targets = dict(Counter(y_test_ext))

#also create train datasets that are external + team data
x_train_ext2 = x_train_ext + x_train
y_train_ext2 = y_train_ext + y_train


#### Get new paramters with new data ####
#Do cross-validation for the Ridge Classifier with the external data
rc_cv_ext = cv_func(pipe = rc_brain, tuned_parameters = rc_parameters,
                 x_train = x_train_ext, y_train = y_train_ext)

#Do cross-validation for the Ridge Classifier with the external data + team data
rc_cv_ext2 = cv_func(pipe = rc_brain, tuned_parameters = rc_parameters,
                 x_train = x_train_ext2, y_train = y_train_ext2)

#Get best parameters of external data
best_params_ext = rc_cv_ext.best_params_
best_ext = best_params_ext['clf__alpha']

#Get mean scores with external data
mean_scores_ext = rc_cv_ext.cv_results_['mean_test_score']
mean_scores_ext = np.array(mean_scores_ext).reshape(len(reg_params), 
                       1)

#Get best parameters of external + team data
#Note: this ends up being the same as the external data! 
best_ext2 = rc_cv_ext2.best_params_['clf__alpha']

#plot results using cv_plot function
cv_plot(rc_cv_ext.cv_results_,
        reg_params, ['na'],
        xlabel = 'Regularization', 
        main_title = 'Ridge Classifier')
cv_plot(rc_cv_ext2.cv_results_,
        reg_params, ['na'],
        xlabel = 'Regularization', 
        main_title = 'Ridge Classifier')
plt.legend(('External Data Only', 'External Data \n + Proj01 Data'),
           bbox_to_anchor = (1.0, 1.01))
plt.plot([best_ext, ] * 2,
         [0.925, max(mean_scores_ext[:, 0])],
         linestyle = '--', color = 'black')

#plot external vs just team data cv to see difference in accuracies
cv_plot(rc_cv_ext.cv_results_,
        reg_params, ['na'],
        xlabel = 'Regularization', 
        main_title = 'Ridge Classifier')
plt.plot([best_ext, ] * 2,
         [0.750, max(mean_scores_ext[:, 0])],
         linestyle = '--', color = 'blue')
cv_plot(rc_cv.cv_results_,
        reg_params, ['na'],
        xlabel = 'Regularization', 
        main_title = 'Ridge Classifier')
plt.plot([reg_params[best_1], ] * 2,
         [0.750, max(mean_scores[:, 0])],
         linestyle = '--', color = 'orange')
plt.legend(('External Data', 'Proj01 Data'),
           bbox_to_anchor = (1.0, 1.01))
plt.show()


#### Re-train Brain ####
#Now that we have the best parameters, re-train brain with different datasets
rc_brain_ext = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RidgeClassifier(alpha = best_params_ext['clf__alpha'])),
                                     ])

#With external data
rc_pred_ext, rc_ext = final_metrics(rc_brain_ext, 5,
                              x_train_ext, y_train_ext, 
                              x_test_ext, y_test_ext)
#with external data + team data
rc_pred_ext2, rc_ext2 = final_metrics(rc_brain_ext, 6,
                              x_train_ext2, y_train_ext2, 
                              x_test_ext, y_test_ext)
#with just team data but new brain
rc_pred2, rc2 = final_metrics(rc_brain_ext, 4,
                              x_train, y_train, x_test_ext, y_test_ext)

#### Get wrong predictions from each brain ####
#just team-data
rc_wrong2 = list()
for p in range(len(rc_pred2)):
    if rc_pred2[p] != y_test_ext[p]:
        print(x_test_ext[p], y_test_ext[p], rc_pred2[p])
        rc_wrong2.append(p)
        
#Just external data
rc_wrong_ext = list()
for p in range(len(rc_pred_ext)):
    if rc_pred_ext[p] != y_test_ext[p]:
        print(x_test_ext[p], y_test_ext[p], rc_pred_ext[p])
        rc_wrong_ext.append(p)
        
#External + team data
rc_wrong_ext2 = list()
for p in range(len(rc_pred_ext2)):
    if rc_pred_ext2[p] != y_test_ext[p]:
        print(x_test_ext[p], y_test_ext[p], rc_pred_ext2[p])
        rc_wrong_ext2.append(p)
        
        
#Get all docs (by index) that are misclassified
ext_wrong = rc_wrong2 + rc_wrong_ext + rc_wrong_ext2
ext_wrong_c = dict(Counter(ext_wrong))

#Get docs that are misclassified by all methods
idx_ext = [i for i, c in ext_wrong_c.items() if c == 3]
docs_ext = [x_test_ext[i] for i in idx_ext]

#Docs that are misclassified by our best two classifiers
docs_rc_ext = [x_test_ext[i] for i in rc_wrong_ext]
docs_rc_ext2 = [x_test_ext[i] for i in rc_wrong_ext2]

#Get a count of targets that are misclassified
targets_ext = [y_test_ext[i] for i in ext_wrong_c.keys()]
targets_ext_c = dict(Counter(targets_ext))

targets_ext_all = [y_test_ext[i] for i in idx_ext]
targets_ext_all_c = dict(Counter(targets_ext_all))



## plot the confusion matricies but normalized this time around
#External data
metrics.plot_confusion_matrix(rc_brain_ext.fit(x_train_ext, y_train_ext),
                              x_test_ext, y_test_ext,
                              normalize = 'true')
plt.title('External Data')

#team-data
metrics.plot_confusion_matrix(rc_brain_ext.fit(x_train, y_train),
                              x_test_ext, y_test_ext,
                              normalize = 'true')
plt.title('Proje01 Data')

#external + team data
metrics.plot_confusion_matrix(rc_brain_ext.fit(x_train_ext2, y_train_ext2),
                              x_test_ext, y_test_ext,
                              normalize = 'true')
plt.title('External Data + Proj01 Data')


#### Compare accuracies of all brains ####
ext_models = np.concatenate((rc, rc_ext, rc_ext2), axis = 0)

ext_models = np.array( sorted(ext_models, key = lambda x:x[1]) )
y_pos = np.arange(3)

#Get bar chart of accuracies
plt.barh(y_pos, ext_models[:, 1], label = 'Accuracy')
plt.yticks(y_pos,
           ['Proj01 Data',
            'External Data Only',
            'External Data +  \n Proj01 Data'])
plt.xlabel('Accuracy')
for i, v in enumerate(ext_models[:, 1]):
    print(i)
    print(v)
    plt.text(v - 0.09 , i - 0.08, str(round(v,3)), color = 'white',
             fontweight = 'bold')

plt.show()

    

## Plot training and testing time using project 01 data
width = 0.4
plt.barh(y_pos, ext_models[:, 3], width, label = 'Testing Time',
         color = 'orange')
plt.barh(y_pos + width, ext_models[:, 2], width, label = 'Training Time',
         color = 'purple')
plt.yticks(y_pos,
           ['Proj01 Data',
            'External Data Only',
            'External Data +  \n Proj01 Data'])
plt.legend()
plt.xlabel('Time (seconds)')
plt.show()

y_test_count = dict(Counter(y_test_ext))






