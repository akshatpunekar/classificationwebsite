from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('./media/school_grades_dataset.csv')


obj_df = df.select_dtypes(include=['object']).copy()
int_df = df.select_dtypes(include=['int64']).copy()

float_df=df.select_dtypes(include=['float64']).copy()

df_int_float = pd.concat([float_df,int_df], axis=1).reindex(int_df.index)

le = preprocessing.LabelEncoder()
obj_df_trf=obj_df.astype(str).apply(le.fit_transform)

df_final = pd.concat([df_int_float,obj_df_trf], axis=1).reindex(df_int_float.index)


labels = df_final.loc[:,'G1']               #Corrected
features = df_final.loc[:,['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic']]


#Training the model
model = OneVsRestClassifier(GaussianNB())
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=42)
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

print(accuracy)
