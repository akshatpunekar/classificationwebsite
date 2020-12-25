from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.contrib import messages

#Importing functions required for pre-processing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#####################
import matplotlib
matplotlib.use('Agg')
#####################

#Importing all the functions required for classification
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier

#Pandas-profiling feature
from pandas_profiling import ProfileReport

#Importing classification metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Importing classifier models
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

# Create your views here.
def home(request):
    return render(request, 'home.html', {'name':'Akshat'})

def detail(request):
    return render(request, 'output.html')

def add(request):
    val1 = request.POST['num1']
    val2 = request.POST['num2']
    res= int(val1) + int(val2)
    return render(request, 'result.html', {'result':res})

def upload(request):
    context = {}
    if request.method == "POST":
        try:
            uploaded_file = request.FILES['document']
            temp = str(uploaded_file)
        except:
            messages.info(request, 'Please upload a file')
            return redirect('upload')


        file_type = temp.split('.')[-1]
        print(file_type)
        if file_type != 'csv' and file_type!='txt' and file_type!='xls' and file_type!='xlsx':
            messages.info(request, 'Upload only excel, csv or text file')
            return redirect('upload')
        else:
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            context['url'] = fs.url(name)
            path_of_file = str('.') + str(fs.url(name))

            if file_type != 'csv' or file_type != 'txt':
                dataframe = pd.read_csv(path_of_file)
            elif file_type!='xls' or file_type!='xlsx':
                dataframe = pd.read_excel(path_of_file)

            parameters = dataframe.columns

            context['parameters'] = parameters


    return render(request, 'upload.html', context)

def mid(request):
    query = str('.')+request.GET.get('filepath')

    file_type = query.split('.')[-1]

    if file_type == 'csv' :
        dataframe = pd.read_csv(query)
    elif file_type == 'txt':
        dataframe = pd.read_csv(query, sep='\t')
    elif file_type == 'xls' or file_type == 'xlsx':
        dataframe = pd.read_excel(query)

    parameters = dataframe.columns

    return render(request, 'mid.html', {'parameters': parameters, 'query':query, 'final_variable': parameters[-1] } )

def check(request):
    query = str(request.GET.get('filepath'))
    method_ = str(request.POST['classification'])
    predictions_wanted = int(request.POST['option'])

    target_variable = str(request.POST['target_variable'])

    testsize = request.POST['split']

    if testsize == '':
        testsize = 0.3

    testsize = float(testsize)

    if testsize>1 or testsize<0:
        testsize = 0.1

    if method_ == 'naive_bayes':
        # Training the model
        model = OneVsRestClassifier(GaussianNB())
        model_name = 'Naive Bayes'
    elif method_ == 'log_reg':
        model = OneVsRestClassifier(LogisticRegression())
        model_name = 'Logistic Regression'
    elif method_ == 'dec_trs':
        model = OneVsRestClassifier(DecisionTreeClassifier())
        model_name = 'Decision Trees'
    elif method_ == 'svm':
        model = OneVsRestClassifier(LinearSVC())
        model_name = 'Support Vector Machines'
    elif method_ == 'knn':
        model = OneVsRestClassifier(KNeighborsClassifier())
        model_name = 'K Nearest Neighbors'
    elif method_ == 'lda':
        model = OneVsRestClassifier(LinearDiscriminantAnalysis())
        model_name = 'Linear Discriminant Analysis'
    elif method_ == 'bdt':
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                  n_estimators=500,
                                  random_state=8)
        model_name = 'Bagging Decision Trees'

    if query.split('.')[-1] == 'txt':
        df = pd.read_csv(query, sep='\t')
    else:
        df = pd.read_csv(query)

    prof = ProfileReport(df.sample(n=10000,replace=True))
    prof.to_file(output_file='./templates/output.html')

    df.dropna()


    # Iterations to read all the features required for classfication
    features = list()

    for column in df.columns:
        try:
            feature = str(request.POST[column])
            features.append(feature)
        except:
            continue
    #End of iterations
    #print(features)

    ########################################

    obj_df = df.select_dtypes(include=['object']).copy()
    int_df = df.select_dtypes(include=['int64']).copy()

    float_df = df.select_dtypes(include=['float64']).copy()

    df_int_float = pd.concat([float_df, int_df], axis=1).reindex(int_df.index)

    le = preprocessing.LabelEncoder()
    obj_df_trf = obj_df.astype(str).apply(le.fit_transform)

    df_final = pd.concat([df_int_float, obj_df_trf], axis=1).reindex(df_int_float.index)

    ##########################################

    labels = df_final.loc[:,target_variable]               #Corrected
    features = df_final.loc[:,features]                    #Corrected


    #Training the model
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=testsize, random_state=42)

    model.fit(train, train_labels)

    # Make predictions
    preds = model.predict(test)


    pred_table = list()
    pred_table.append(['Actual Value','Predicted value'])

    for value in range(len(preds)):
        pred_table.append([test_labels.tolist()[value],preds[value]])

    # Evaluate accuracy
    accuracy = accuracy_score(test_labels, preds) * 100
    cm = confusion_matrix(test_labels, preds)
    cnf_matrix = cm.tolist()

    #################################################################
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    #################################################################

    #Calculating required metrics
    kappa_value = cohen_kappa_score(test_labels.tolist(), preds)
    matthew_value = matthews_corrcoef(test_labels.tolist(), preds)
    precision = precision_score(test_labels.tolist(), preds, average='weighted')
    recall = recall_score(test_labels.tolist(), preds, average='weighted')

    return render(request, 'check.html', {'query' : query, 'table': cnf_matrix,
                                          'acc':accuracy, 'model_name':model_name, 'finaltable':pred_table,
                                          'required':predictions_wanted,
                                          'kappa':kappa_value, 'matthew':matthew_value,
                                          'tpr':TPR, 'fpr':FPR,
                                          'precision':precision, 'recall':recall})