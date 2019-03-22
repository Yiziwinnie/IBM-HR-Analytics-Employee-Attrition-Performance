#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:37:25 2018

@author: yizihuang

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, chi2
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from tabulate import tabulate
from math import sqrt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import mean_squared_error


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import math 
from math import sqrt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
from sklearn.svm import SVC
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, chi2
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from tabulate import tabulate
from math import sqrt





def performance_DT1(X_train,y_train,X_test,y_test):  
    Sensitivity_train_list=[]
    Specificity_train_list=[]
    Sensitivity_test_list=[]
    Specificity_test_list=[] 
    seed= random.sample(range(100),20)
    for i in seed:
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = i , max_depth=20, min_samples_split=10, min_samples_leaf= 5)
        clf_gini.fit(X_train, y_train)
        y_train_pred_gini = clf_gini.predict(X_train)
        y_pred_gini = clf_gini.predict(X_test)


        ## Train
        tn1, fp1, fn1, tp1 = confusion_matrix(y_train, y_train_pred_gini).ravel()
        Sensitivity_train = tp1/(tp1+fn1)
        Specificity_train = tn1/(tn1+fp1)
        Sensitivity_train_list.append(Sensitivity_train)
        Specificity_train_list.append(Specificity_train)

        ## Test
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_gini).ravel()
        Sensitivity_test = tp2/(tp2+fn2)
        Specificity_test = tn2/(tn2+fp2)
        Sensitivity_test_list.append(Sensitivity_test)
        Specificity_test_list.append(Specificity_test)

    print("Sensitivity_train:" +str(sum(Sensitivity_train_list)/len(Sensitivity_train_list)))
    print("Specificity_train:" +str(sum(Specificity_train_list)/len(Specificity_train_list)))
    print("Sensitivity_test:"  +str(sum(Sensitivity_test_list)/len(Sensitivity_test_list)))
    print("Specificity_test:"  +str(sum(Specificity_test_list)/len(Specificity_test_list)))
    
    

os.getcwd()
os.chdir("/Users/yizihuang/Documents/CSC540/Final Project")

data=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

X = data.drop(columns="Attrition")
y = data["Attrition"]

y.value_counts().plot(kind="bar")
plt.title("Class Distribution", fontsize=17)
plt.xlabel(" Type of status", fontsize=13)
plt.ylabel("Count",fontsize=13)



data1= data.copy()

cleanup_nums = {
"BusinessTravel": {"Non-Travel":1, "Travel_Rarely" :2,"Travel_Frequently":3},
"Department": {"Research & Development":1,"Sales":2,"Human Resources":3},
"EducationField": {"Life Sciences":1,"Medical":2,"Marketing":3,"Technical Degree":4,"Human Resources":5,"Other":6},
"Gender": {"Male":1,"Female":2},
"JobRole": {"Sales Executive":1,"Research Scientist":2, "Laboratory Technician":3,"Manufacturing Director":4,"Healthcare Representative":5,"Manager":6,"Sales Representative":7,"Research Director":8,"Human Resources":9},
"MaritalStatus": {"Single":1, "Married":2, "Divorced":3},
"OverTime": {"No":1,"Yes":2},
"Attrition" :{"No":0,"Yes":1}
}



ax = sns.violinplot(x="Gender", y="YearsSinceLastPromotion", hue="Attrition",data=data, palette="Set2", split=True,scale="count", inner="quartile")
ax = sns.violinplot(x="Gender", y="YearsSinceLastPromotion", hue="Attrition",data=data_new, palette="Set2", split=True,scale="count", inner="quartile")



## Check the missing value
data1.info()
data1.isnull().values.any()
data1.describe()


## Check the distribution of classes : bar graph, volin plot, 


## Confusion Matrix : Plot a correlation map for all numeric variables
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


#### split numeric + categorical
numercic_data= data1.select_dtypes(exclude=['object'])
categorical_data= data1.select_dtypes(include=['object'])


reorder_categorical_data = pd.concat([categorical_data.iloc[:,1:],categorical_data["Attrition"]],axis=1)
data_new= pd.concat([numercic_data,reorder_categorical_data],axis=1)
data_new.replace(cleanup_nums, inplace=True)

numeric_data_norm= preprocessing.normalize(data_new.iloc[:,:24])
numeric_data_norm=pd.DataFrame(numeric_data_norm)

numeric_data_norm.columns=data_new.iloc[:,:24].columns

numer_cate_data=pd.concat([numeric_data_norm,data_new.iloc[:,24:]],axis=1)

X = data_new.drop(columns="Attrition")
#lb = preprocessing.LabelBinarizer()
#y = lb.fit_transform(data_new['Attrition'])
y=data_new['Attrition']



X_split = numer_cate_data.iloc[:,:-1]
y_split = numer_cate_data.iloc[:,-1]


## Oversampling  X_train_over = 1974
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=12)

ros = RandomOverSampler(random_state=12)
X_train_over, y_train_over = ros.fit_sample(X_train, y_train)


pd.Series(y_train_over).value_counts().plot(kind="bar")
plt.title("Class Distribution: After Oversampling", fontsize=17)
plt.xlabel(" Type of status", fontsize=13)
plt.ylabel("Count",fontsize=13)




## Under sampling  X_train_under= 378
res = RandomUnderSampler(random_state=12)
X_train_under, y_train_under = res.fit_sample(X_train, y_train)



pd.Series(y_train_under).value_counts().plot(kind="bar")
plt.title("Class Distribution: After Undersampling", fontsize=17)
plt.xlabel(" Type of status", fontsize=13)
plt.ylabel("Count",fontsize=13)



## Normalize
X_norm = numer_cate_data.iloc[:,:-1]
y = numer_cate_data.iloc[:,-1]


X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y, test_size = 0.2,random_state=12)

"""
data_X_norm = preprocessing.normalize(data_new.drop(columns="Attrition"))
data_X_norm= pd.DataFrame(data_X_norm )
data_norm= pd.concat([data_X_norm,data_new['Attrition']], axis=1)
X_norm=data_norm.iloc[:,:-1]
y=data_norm.iloc[:,-1]"""



## Normalize + Oversampling
X_train_norm_over, y_train_norm_over= ros.fit_sample(X_train_norm, y_train_norm)


## Normalize +undersampling 
X_train_norm_under, y_train_norm_under = res.fit_sample(X_train_norm, y_train_norm)


## Oversampling + Normalize   len(X_train_d)=1974
X_train_over, y_train_over = ros.fit_sample(X_train, y_train)
X_train_over_d=pd.DataFrame(X_train_over)


X_test.columns=list(range(0,31))
X_over = pd.concat([X_train_over_d, X_test],axis=0)


pre1=pd.DataFrame(preprocessing.normalize(X_over.iloc[:,:24]))
pre2=pd.DataFrame(preprocessing.normalize(X_over.iloc[:,24:]))
X_over_norm = pd.concat([pre1,pre2],axis=1)

X_train_over_norm=X_over_norm[:1974]
X_test_over_norm=X_over_norm[1974:]


#plot_Roc(X_train_over_norm, X_test_over_norm, y_train_over, y_test, DecisionTreeClassifier)


## Undersampling + Normalize   len(X_train_under)=378 
X_train_under, y_train_under = res.fit_sample(X_train, y_train)
X_train_under_d=pd.DataFrame(X_train_under)

X_test.columns=list(range(0,31))
X_under = pd.concat([X_train_under_d, X_test],axis=0)



pre3=pd.DataFrame(preprocessing.normalize(X_under.iloc[:,:24]))
pre4=pd.DataFrame(preprocessing.normalize(X_under.iloc[:,24:]))
X_under_norm = pd.concat([pre3,pre4],axis=1)

X_train_under_norm = X_under_norm[:378]
X_test_under_norm  = X_under_norm[378:]



#plot_Roc(X_train_under_norm, X_test_under_norm, y_train_under, y_test, DecisionTreeClassifier)



## Feaure Extraction
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_new.drop(columns="Attrition"))
print(pca.explained_variance_ratio_)
print(round(pca.explained_variance_ratio_.sum(),2))

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split( principalComponents, data_new["Attrition"], test_size = 0.2, random_state = 12)

## PCA + over sampling
X_train_pca_over, y_train_pca_over = ros.fit_sample(X_train_pca, y_train_pca)


#plot_Roc(X_train_pca_over, X_test_pca, y_train_pca_over, y_test_pca,DecisionTreeClassifier)



## PCA + under sampling
X_train_pca_under, y_train_pca_under = res.fit_sample(X_train_pca, y_train_pca)

#plot_Roc(X_train_pca_under, X_test_pca, y_train_pca_under, y_test_pca,DecisionTreeClassifier)



## over sampling + normalize + PCA  X_train= 1176 X_train(over)=1974 X_test=294 X=2268

pca2 = PCA(n_components=2)
principalComponents2 = pca2.fit_transform(X_over_norm)
print(round(pca2.explained_variance_ratio_.sum(),2))


X_over_norm_pca= pd.DataFrame(principalComponents2)

X_train_over_norm_pca=X_over_norm_pca[:1974]
X_test_over_norm_pca =X_over_norm_pca[1974:]

#plot_Roc(X_train_over_norm_pca, X_test_over_norm_pca, y_train_over, y_test ,DecisionTreeClassifier)


## under sampling + normalize + feature extraction  X_train= 378  X_test=294  X= 672

pca3 = PCA(n_components=7)
principalComponents3 = pca3.fit_transform(X_under_norm)
print(round(pca3.explained_variance_ratio_.sum(),2))

X_under_norm_pca= pd.DataFrame(principalComponents3)

X_train_under_norm_pca=X_under_norm_pca[:378]
X_test_under_norm_pca =X_under_norm_pca[378:]

#plot_Roc(X_train_under_norm_pca, X_test_under_norm_pca, y_train_under, y_test ,DecisionTreeClassifier)



## over sampling + PCA  X_train= 1176 X_train(over)=1974 X_test=294 X=2268

pca4 = PCA(n_components=2)
principalComponents4 = pca4.fit_transform(X_over)
print(round(pca4.explained_variance_ratio_.sum(),2))


X_over_pca= pd.DataFrame(principalComponents4)

X_train_over_pca=X_over_pca[:1974]
X_test_over_pca =X_over_pca[1974:]

#plot_Roc(X_train_over_pca, X_test_over_pca, y_train_over, y_test ,DecisionTreeClassifier)




## under sampling + feature extraction  X_train= 378  X_test=294  X= 672

pca5 = PCA(n_components=2)
principalComponents5 = pca5.fit_transform(X_under)
print(round(pca5.explained_variance_ratio_.sum(),2))

X_under_pca= pd.DataFrame(principalComponents5)

X_train_under_pca=X_under_pca[:378]
X_test_under_pca =X_under_pca[378:]

# plot_Roc(X_train_under_pca, X_test_under_pca, y_train_under, y_test ,DecisionTreeClassifier)




## Feature selection 
X_train, X_test, y_train, y_test = train_test_split(data_new.drop(columns="Attrition"), data_new['Attrition'], test_size = 0.2,random_state=12)

selector = SelectKBest(chi2, k="all")
X_new = selector.fit_transform(X_train, y_train)
names = data_new.drop(columns="Attrition").columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])



x_cord=list(range(1,32)) 
X_train_feature_list=[]
X_test_feature_list=[]
for i in x_cord:
    topfeature_train= X_train[ns_df_sorted["Feat_names"][:i,]]
    X_train_feature_list.append(topfeature_train)
    
    topfeature_test= X_test[ns_df_sorted["Feat_names"][:i,]]
    X_test_feature_list.append(topfeature_test)
      
performance=[]
for i, j in zip(X_train_feature_list,X_test_feature_list):
    p=performance_DT1(i,y_train,j,y_test)
    performance.append(p)


i=10
X_train_select= X_train[ns_df_sorted["Feat_names"][:i,]]
X_test_select= X_test[ns_df_sorted["Feat_names"][:i,]]





#plot_Roc(X_train_select, X_test_select, y_train, y_test ,DecisionTreeClassifier)


## over sampling + normalization + feature selection   X_train= 1176 X_train(over)=1974 X_test=294 X=2268
X_over_norm_d=pd.DataFrame(X_over_norm)
X_over_norm_d.columns=X_train.columns 


X_train_over_norm_select = X_over_norm_d[ns_df_sorted["Feat_names"][:i,]][:1974]
X_test_over_norm_select =  X_over_norm_d[ns_df_sorted["Feat_names"][:i,]][1974:]

#plot_Roc(X_train_over_norm_select, X_test_over_norm_select, y_train_over, y_test ,DecisionTreeClassifier)



## under dampling  + normalization + feature selection X_train= 378  X_test=294  X= 672

X_under_norm_d=pd.DataFrame(X_under_norm)
X_under_norm_d.columns=X_train.columns 


X_train_under_norm_select = X_under_norm_d[ns_df_sorted["Feat_names"][:i,]][:378]
X_test_under_norm_select =  X_under_norm_d[ns_df_sorted["Feat_names"][:i,]][378:]

#plot_Roc(X_train_under_norm_select, X_test_under_norm_select, y_train_under, y_test ,DecisionTreeClassifier)


## over sampling  + feature selection   X_train= 1176 X_train(over)=1974 X_test=294 X=2268

X_over.columns=X_train.columns

X_train_over_select = X_over[ns_df_sorted["Feat_names"][:i,]][:1974]
X_test_over_select =  X_over[ns_df_sorted["Feat_names"][:i,]][1974:]


#plot_Roc(X_train_over_select, X_test_over_select, y_train_over, y_test ,DecisionTreeClassifier)



## under sampling  + feature selection X_train= 378  X_test=294  X= 672

X_under.columns=X_train.columns


X_train_under_select = X_under[ns_df_sorted["Feat_names"][:i,]][:378]
X_test_under_select =  X_under[ns_df_sorted["Feat_names"][:i,]][378:]

#plot_Roc(X_train_under_select, X_test_under_select, y_train_under, y_test ,DecisionTreeClassifier)



### Find the optimal point 

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds,fpr,tpr))
    thred_new =j_ordered[-1][1]
    fpr_new =j_ordered[-1][2]
    tpr_new = j_ordered[-1][3]
    
    return (fpr_new,tpr_new,thred_new)


### Performance including ROC,AUC,Accuracy, Sensitivity, Spesificity

def SVM_performance(X_train,X_test,y_train,y_test):
    svm = LinearSVC(random_state=111,C=1)
    clf = CalibratedClassifierCV(svm) 
    clf.fit(X_train, y_train)

    preds1 = clf.predict_proba(X_train)[::,1]
    pred_train= clf.predict(X_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, preds1)
    thresholds = np.delete(thresholds, 0)
    fpr = np.delete(fpr, 0)
    tpr = np.delete(tpr, 0)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC:SVM_Training data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='green')
    ax2.set_ylabel('Threshold',color='green')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
    plt.show()

    preds2 = clf.predict_proba(X_test)[::,1]
    pred= clf.predict(X_test)

    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, preds2)
    thresholds2 = np.delete(thresholds2, 0)
    fpr2 = np.delete(fpr2, 0)
    tpr2 = np.delete(tpr2, 0)
    roc_auc2 = auc(fpr2, tpr2)
    
    ##set the threshold that can max the tpr & fpr 

    fpr_new,tpr_new,thred_new= cutoff_youdens_j(fpr2,tpr2,thresholds2)

    
    #plt.title('ROC:SVM_Test data')
    plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2)
    plt.plot(fpr_new, tpr_new, 'bo')
    #plt.plot([fpr_new, fpr_new, 0], [0, tpr_new, tpr_new], 'k-', lw=1,dashes=[2, 2])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    ax2 = plt.gca().twinx()
    ax2.plot(fpr2, thresholds2, markeredgecolor='r',linestyle='dashed', color='green')
    ax2.set_ylabel('Threshold',color='green')
    ax2.set_ylim([thresholds2[-1],thresholds2[0]])
    ax2.set_xlim([fpr2[0],fpr2[-1]])
    plt.show()



    predict_mine = np.where(preds2 >=thred_new, 1, 0)
    #pred_train_new = np.where(preds1 >=thred_new, 1, 0)

    ## get confusion matrix

    cm_test = confusion_matrix(y_test, predict_mine) 
    cm_train = confusion_matrix(y_train, pred_train) 
    tn1, fp1, fn1, tp1 = confusion_matrix(y_test, predict_mine).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(y_train, pred_train).ravel()


    ## optimal test
    Sensitivity_test = tp1/(tp1+fn1)
    Specificity_test = tn1/(tn1+fp1)
    Performance_test = Sensitivity_test+Specificity_test 

    ## train
    Sensitivity_train = tp2/(tp2+fn2)
    Specificity_train= tn2/(tn2+fp2)
    Performance_train =Sensitivity_train+Specificity_train

    Accuracy_test = round(accuracy_score(y_test, predict_mine),2)
    Accuracy_train = round(accuracy_score(y_train,pred_train),2)



    table = tabulate([
            ['Threshold',thred_new],\
            ['Sensitivity_train',Sensitivity_train],\
            ['Specificity_train', Specificity_train],\
            ['Performance_train', Performance_train],\
            ['Sensitivity_test',Sensitivity_test],\
            ['Specificity_test',Specificity_test],\
            ['Performance_test',Performance_test],\
            ['Accuracy_train', Accuracy_train],\
            ['Accuracy_test', Accuracy_test]],headers=['Performance', 'Value'], tablefmt='orgtbl')

    comparison_table=pd.concat([pd.DataFrame(preds2[:10]),pd.DataFrame(pred[:10]),pd.DataFrame(predict_mine[:10])],axis=1)
    comparison_table.columns=['Probability','Prediction','Optimal Prediction']
    
    print ("Confusion Matrix_train:{}".format(cm_train))
    print ("Confusion Matrix_test :{}".format(cm_test))
    print(table)
    print(comparison_table)
    return cm_test


SVM_performance(X_train,X_test,y_train,y_test)


SVM_performance(X_train_over, X_test, y_train_over, y_test)
SVM_performance(X_train_under, X_test, y_train_under, y_test)


SVM_performance(X_train_norm, X_test_norm, y_train_norm, y_test_norm)


SVM_performance(X_train_norm_over, X_test_norm, y_train_norm_over, y_test_norm)
SVM_performance(X_train_norm_under, X_test_norm, y_train_norm_under, y_test_norm)


SVM_performance(X_train_over_norm, X_test_over_norm, y_train_over, y_test)
SVM_performance(X_train_under_norm, X_test_under_norm, y_train_under, y_test)


SVM_performance(X_train_pca, X_test_pca, y_train_pca, y_test_pca)


SVM_performance(X_train_pca_over, X_test_pca, y_train_pca_over, y_test_pca)
SVM_performance(X_train_pca_under, X_test_pca, y_train_pca_under, y_test_pca)


SVM_performance(X_train_over_pca, X_test_over_pca, y_train_over, y_test)
SVM_performance(X_train_under_pca, X_test_under_pca, y_train_under, y_test)

SVM_performance(X_train_over_norm_pca, X_test_over_norm_pca, y_train_over, y_test)
SVM_performance(X_train_under_norm_pca, X_test_under_norm_pca, y_train_under, y_test)


SVM_performance(X_train_select, X_test_select, y_train, y_test)


SVM_performance(X_train_over_select, X_test_over_select, y_train_over, y_test)
SVM_performance(X_train_under_select, X_test_under_select, y_train_under, y_test)

SVM_performance(X_train_over_norm_select, X_test_over_norm_select, y_train_over, y_test)
SVM_performance(X_train_under_norm_select, X_test_under_norm_select, y_train_under, y_test)


   

res = RandomUnderSampler(random_state=12)
X_train_under, y_train_under = res.fit_sample(X_train, y_train)


def ploterror_rate_C(X_train,X_test,y_train,y_test):
    seed= range(20)
    C =[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    avg_err_rate_train_list=[]
    avg_err_rate_test_list=[]
    for c in C:
        error_rate_train_list=[]
        error_rate_test_list=[]
        for i in seed:
            clf = LinearSVC(C=c,random_state = i)
            clf.fit(X_train, y_train)
            y_pred= clf.predict(X_test)
            y_train_pred = clf.predict(X_train)     
            resubstutution_B = 1- accuracy_score(y_train,y_train_pred)
            generalization_B = 1- accuracy_score(y_test, y_pred)
            error_rate_train_list.append(resubstutution_B) 
            error_rate_test_list.append(generalization_B)
        avg_err_rate_train_list.append(sum(error_rate_train_list)/len(error_rate_train_list))
        avg_err_rate_test_list.append(sum(error_rate_test_list)/len(error_rate_test_list))
    plt.plot( C,avg_err_rate_train_list,label='Error_rate_train',color="blue")
    plt.plot( C,avg_err_rate_train_list,label='Error_rate_test',color="red",alpha=0.5)
    plt.legend()
    plt.ylabel('Error rate')
    plt.xlabel('The value of C')
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ploterror_rate_C(X_train_over_norm, X_test_over_norm, y_train_over, y_test)












