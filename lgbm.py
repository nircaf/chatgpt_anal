from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.metrics import accuracy_score
import seaborn as sns
from pandas_datareader import data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import xgboost
import lightgbm as lgb
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from catboost import *
from sklearn.neural_network import MLPClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
from sklearn.linear_model import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from lazypredict.Supervised import LazyClassifier, LazyRegressor
import pickle
import os
from sklearn.model_selection import GridSearchCV

subplot_num = 3
ranodm_state = 42
n_estimators = 2000

def run_best_model_grid_search(X_train, X_test, y_train, y_test):
    try:
        # load saved model
        pickled_model = pickle.load(open('saved_models/model_exp_.pickle', 'rb'))
    except:
        raise('No saved model found in \n saved_models/model_exp_.pickle')
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

def light_gbm(X_train, X_test, y_train, y_test):
    # train and test split
    model = lgb.LGBMClassifier(learning_rate=0.01, max_depth=-5,
                               random_state=ranodm_state, n_estimators=n_estimators)
    plot_sklearn(model, X_train, X_test, y_train, y_test, 'light_gbm')


def xgboosting(X_train, X_test, y_train, y_test):
    # declare parameters
    params = {
        'objective': 'multi:softmax',
        'max_depth': 5,
        'alpha': 10,
        'learning_rate': 0.01,
        'n_estimators': n_estimators
    }
    # instantiate the classifier
    model = XGBClassifier(**params)
    plot_sklearn(model, X_train, X_test, y_train, y_test, 'XGBoost')


# def regressor_model(X_train, X_test, y_train, y_test):
#     reg = LazyRegressor(verbose=0, custom_metric=None)
#     models, predictions = reg.fit(np.array(X_train,dtype=np.float), np.array(X_test,dtype=np.float)
#         , np.array(y_train,dtype=np.int), np.array(y_test,dtype=np.int))
#     print(models)
#     plot_all_models(models,lazy = True)

def plot_sklearn(model, X_train, X_test, y_train, y_test, title):
    # split X and y into training and testing sets
    model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)],
              verbose=n_estimators/10)
    print('Training accuracy {:.4f}'.format(model.score(X_train, y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(X_test, y_test)))
    #   lgb.plot_importance(model,title=f'plot_importance')
    #   lgb.plot_metric(model,title=f'Plot metric')
    #   lgb.plot_tree(model,figsize=(30,40))
    print(metrics.classification_report(y_test, model.predict(X_test)))

def set_params(x_train, x_test, y_train, y_test,model = None,args= None):
    # model.get_params()
    class_dist = y_train.value_counts().to_dict()
    class_weights = {}
    for c_item in class_dist.items():
        class_weights[c_item[0]] = sum(
            class_dist.values()) / ((x_train.shape[1]) * c_item[1])
    params = {
        'class_weight': class_weights,
    }
    # model.set_params(**params)
    return params

def ensemble_model_all(X_train, X_test, y_train, y_test):
    # Fit all models
    clf = LazyClassifier(verbose=0, custom_metric = None)
    models, predictions = clf.fit(np.array(X_train,dtype=np.float), np.array(X_test,dtype=np.float)
        , np.array(y_train,dtype=np.int), np.array(y_test,dtype=np.int))
    print(models)
    plot_all_models(models,lazy = True)

def ensemble_model(x_train, x_test, y_train, y_test, args = None):
    knn = KNeighborsClassifier()
    svc = LinearSVC()
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    gnb = GaussianNB()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    gbc = GradientBoostingClassifier()
    ada = AdaBoostClassifier()
    # -------------------------------------------------------------------
    models = []
    models.append(('KNeighborsClassifier', knn))
    models.append(('SVC', svc))
    models.append(('LogisticRegression', lr))
    models.append(('DecisionTreeClassifier', dt))
    models.append(('GaussianNB', gnb))
    models.append(('RandomForestClassifier', rfc))
    models.append(('XGBClassifier', xgb))
    models.append(('GradientBoostingClassifier', gbc))
    models.append(('AdaBoostClassifier', ada))
    models.append(('BernoulliNB', BernoulliNB()))

    models.append(('SGDClassifier', SGDClassifier()))
    models.append(('LogisticRegression', LogisticRegression(multi_class="multinomial")))
    models.append(('RidgeClassifier', RidgeClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('MLPClassifier', MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)))
    models.append(('catboost', CatBoostClassifier(
        iterations=5,
        learning_rate=0.1,
        # loss_function='CrossEntropy'
        )))
    # models.append(('LGBMClassifier', lgb.LGBMClassifier(num_leaves=12,
    #                                                     learning_rate=0.01,
    #                                                     n_estimators=5000,
    #                                                     objective='multiclass',
    #                                                     class_weight=class_weights
    #                                                     )))

    # --------------------------------------------------------------------
    Model = []
    score = []
    cv = []
    for name, model in models:
        print('*****************', name, '*******************')
        print('\n')
        Model.append(name)
        model.fit(np.array(x_train,dtype=np.float), np.array(y_train,dtype=np.int))
        print(model)
        pre = model.predict( np.array(x_test,dtype=np.float))
        print('\n')
        AS = accuracy_score( np.array(y_test,dtype=np.int), pre)
        print('Accuracy_score  -', AS)
        score.append(AS*100)
        print('\n')
        sc = cross_val_score(model, x_test, y_test, cv=10,
                             scoring='accuracy').mean()
        print('cross_val_score  -', sc)
        cv.append(sc*100)
        print('\n')
        print('classification report\n', classification_report(np.array(y_test,dtype=np.int), pre))
        print('\n')
        cm = confusion_matrix(np.array(y_test,dtype=np.int), pre)
        print(cm)
        print('\n')
        plt.figure(figsize=(10, 40))
        plt.subplot(911)
        plt.title(f'{name} accuracy cross val: {sc*100}%')
    plot_all_models(Model,cv)
    # save model to file
    cv = np.where(np.isnan(cv), 0, cv) # replace nan with 0
    pickle.dump(models[np.argmax(cv)][-1] , open(os.path.join('saved_models',
        'model_exp'  + '.pickle'), 'wb'))



def plot_all_models(Model,cv= None,lazy=False):
    plt.figure(figsize=(10, 40))
    plt.subplot(911)
    plt.title(f'all models comparison')
    if lazy:
        ax = sns.barplot(x=Model.index,y = Model.iloc[:,2])
        plt.ylabel(Model.columns[2])
    else:
        ax = sns.barplot(x=Model, y=cv)
        plt.ylabel('Accuracy')
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=-90)
    plt.savefig('Nir_figures/plot_all_models.png')


def ensemble_model_grid_search(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier()
    svc = SVC()
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    gnb = GaussianNB()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    gbc = GradientBoostingClassifier()
    ada = AdaBoostClassifier()
    # -------------------------------------------------------------------
    models = []
    models.append(('KNeighborsClassifier', knn))
    models.append(('SVC', svc))
    models.append(('LogisticRegression', lr))
    models.append(('DecisionTreeClassifier', dt))
    models.append(('GaussianNB', gnb))
    models.append(('RandomForestClassifier', rfc))
    models.append(('XGBClassifier', xgb))
    models.append(('GradientBoostingClassifier', gbc))
    models.append(('AdaBoostClassifier', ada))
    models.append(('LGBMClassifier', lgb.LGBMClassifier(num_leaves=12,
                                                        learning_rate=0.01,
                                                        n_estimators=5000,
                                                        objective='multiclass',
                                                        class_weight=class_weights
                                                        )))

    # --------------------------------------------------------------------
    Model = []
    score = []
    cv = []
    for name, model in models:
        print('*****************', name, '*******************')
        print('\n')
        params = set_params(x_train, x_test, y_train, y_test, model)
        Model.append(name)
        model.fit(x_train, y_train)
        print(model)
        pre = model.predict(x_test)
        print('\n')
        AS = accuracy_score(y_test, pre)
        print('Accuracy_score  -', AS)
        score.append(AS*100)
        print('\n')
        sc = cross_val_score(model, x_test, y_test, cv=10,
                             scoring='accuracy').mean()
        print('cross_val_score  -', sc)
        cv.append(sc*100)
        print('\n')
        print('classification report\n', classification_report(y_test, pre))
        print('\n')
        cm = confusion_matrix(y_test, pre)
        print(cm)
        print('\n')
        plt.figure(figsize=(10, 40))
        plt.subplot(911)
        plt.title(f'{name} accuracy cross val: {sc*100}%')
    plt.figure(figsize=(10, 40))
    plt.subplot(911)
    plt.title(f'all models comparison')
    ax = sns.barplot(x=Model, y=cv)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=-15)
