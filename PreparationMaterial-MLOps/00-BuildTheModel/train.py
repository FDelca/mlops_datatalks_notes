import pickle
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Parameters of Logistic Regression Model
C = 0.5
output_file = f'model_C={C}.bin'

# Data Preparation
df = pd.read_csv(os.path.join('WA_Fn-UseC_-Telco-Customer-Churn.csv'))


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values

del df_train['churn']
del df_val['churn']

# Define Features Types

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Training
def train(df, y, C=1.0):
    # cat = df[categorical + numerical].to_dict(orient='rows') # Deprecated
    cat = df[categorical + numerical].to_dict('records')
    
    # DictVectorizer - notice that both categorical and numerical are an input
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict('records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Training the Final Model
y_train = df_train_full.churn.values
y_test = df_test.churn.values

dv, model = train(df_train_full, y_train, C = C)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print('auc = %.3f' % auc)


# Save Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    
print(f'The model is saved to {output_file}')
