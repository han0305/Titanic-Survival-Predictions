import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
from sklearn import metrics

pd.pandas.set_option('display.max_columns', None)
 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('train.csv')
print(data.head())

submission = pd.read_csv('test.csv')
print(submission.head())
print(data.dtypes)

print('Number of PassengerId labels: ', len(data.PassengerId.unique()))
print('Number  of passengers on the Titanic: ', len(data))

categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
print(categorical)

numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))
print(numerical)

print(data[categorical].head())
print(data[numerical].head())

for var in ['Pclass',  'SibSp', 'Parch']:
    print(var, ' values: ', data[var].unique())
    
print(data.isnull().mean())
numerical = [var for var in numerical if var not in['Survived','PassengerId']]
print(numerical)



plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Age.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')
 
plt.subplot(1, 2, 2)
fig = data.Fare.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Fare')

plt.show()

Upper_boundary = data.Age.mean() + 3* data.Age.std()
Lower_boundary = data.Age.mean() - 3* data.Age.std()
print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
 

IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

for var in ['Pclass',  'SibSp', 'Parch']:
    print(data[var].value_counts() / np.float(len(data)))
    print()

for var in categorical:
    print(var, ' contains ', len(data[var].unique()), ' labels')

data['Cabin_numerical'] = data.Cabin.str.extract('(\d+)') 
data['Cabin_numerical'] = data['Cabin_numerical'].astype('float') 
 
data['Cabin_categorical'] = data['Cabin'].str[0] 
 

submission['Cabin_numerical'] = submission.Cabin.str.extract('(\d+)')
submission['Cabin_numerical'] = submission['Cabin_numerical'].astype('float')
 
submission['Cabin_categorical'] = submission['Cabin'].str[0]
 
print(data[['Cabin', 'Cabin_numerical', 'Cabin_categorical']].head())

data.drop(labels='Cabin',inplace=True,axis=1)
submission.drop(labels='Cabin',inplace=True,axis=1)

data['Ticket_numerical']=data.Ticket.apply(lambda s: s.split()[-1])
data['Ticket_numerical']=np.where(data.Ticket_numerical.str.isdigit(),data.Ticket_numerical,np.nan)
data['Ticket_numerical'] = data['Ticket_numerical'].astype('float')

data['Ticket_categorical'] = data.Ticket.apply(lambda s: s.split()[0])
data['Ticket_categorical'] = np.where(data.Ticket_categorical.str.isdigit(), np.nan, data.Ticket_categorical)
 

submission['Ticket_numerical'] = submission.Ticket.apply(lambda s: s.split()[-1])
submission['Ticket_numerical'] = np.where(submission.Ticket_numerical.str.isdigit(), submission.Ticket_numerical, np.nan)
submission['Ticket_numerical'] = submission['Ticket_numerical'].astype('float')
 
submission['Ticket_categorical'] = submission.Ticket.apply(lambda s: s.split()[0])
submission['Ticket_categorical'] = np.where(submission.Ticket_categorical.str.isdigit(), np.nan, submission.Ticket_categorical)
 
data[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()

print(data.Ticket_categorical.unique())

text = data.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
print(pd.concat([text, data.Ticket_categorical], axis=1))

text = text.str.upper()
print(text.unique())

data['Ticket_categorical'] = text
 
submission['Ticket_categorical'] = submission.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
submission['Ticket_categorical'] = submission['Ticket_categorical'].str.upper()

data.drop(labels='Ticket', inplace=True, axis=1)
submission.drop(labels='Ticket', inplace=True, axis=1)

def get_title(passenger):
    
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
data['Title'] = data['Name'].apply(get_title)
submission['Title'] = submission['Name'].apply(get_title)
 
print(data[['Name', 'Title']].head())

data.drop(labels='Name', inplace=True, axis=1)
submission.drop(labels='Name', inplace=True, axis=1)

data['Family_size'] = data['SibSp']+data['Parch']+1
submission['Family_size'] = submission['SibSp']+submission['Parch']+1
 
print(data.Family_size.value_counts()/ np.float(len(data)))
 
(data.Family_size.value_counts() / np.float(len(data))).plot.bar()

data['is_mother'] = np.where((data.Sex =='female')&(data.Parch>=1)&(data.Age>18),1,0)
submission['is_mother'] = np.where((submission.Sex =='female')&(submission.Parch>=1)&(submission.Age>18),1,0)
 
print(data[['Sex', 'Parch', 'Age', 'is_mother']].head())

print(data.loc[data.is_mother==1, ['Sex', 'Parch', 'Age', 'is_mother']].head())

print(data[['Cabin_numerical', 'Ticket_numerical', 'is_mother', 'Family_size']].isnull().mean())

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Cabin_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Cabin number')
 
plt.subplot(1, 2, 2)
fig = data.Ticket_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Ticket number')

plt.show()


IQR = data.Ticket_numerical.quantile(0.75) - data.Ticket_numerical.quantile(0.25)
Lower_fence = data.Ticket_numerical.quantile(0.25) - (IQR * 3)
Upper_fence = data.Ticket_numerical.quantile(0.75) + (IQR * 3)
print('Ticket number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
passengers = len(data[data.Ticket_numerical>Upper_fence]) / np.float(len(data))
print('Number of passengers with ticket values higher than {upperboundary}: {passengers}'.format(upperboundary=Upper_fence,passengers=passengers))

print(data[['Cabin_categorical', 'Ticket_categorical', 'Title']].isnull().mean())

for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(var, ' contains ', len(data[var].unique()), ' labels')
for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(data[var].value_counts() / np.float(len(data)))
    print()

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.2,random_state=0)
print(X_train.shape, X_test.shape)   

 
def find_categorical_and_numerical_variables(dataframe):
    cat_vars = [col for col in data.columns if data[col].dtypes == 'O']
    num_vars  = [col for col in data.columns if data[col].dtypes != 'O']
    return cat_vars, num_vars
                 
categorical, numerical = find_categorical_and_numerical_variables(data)
print(categorical)
print(numerical) 

numerical = [var for var in numerical if var not in ['Survived','PassengerId']]
print(numerical)

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
        
def impute_na(X_train, df, variable):
    
    temp = df.copy()
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0)
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]

for df in [X_train, X_test, submission]:
    for var in ['Age', 'Ticket_numerical']:
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    

for df in [X_train, X_test, submission]:
    for var in ['Age', 'Ticket_numerical']:
        df[var] = impute_na(X_train, df, var)
    
 

extreme = X_train.Cabin_numerical.mean() + X_train.Cabin_numerical.std()*3
for df in [X_train, X_test, submission]:
    df.Cabin_numerical.fillna(extreme, inplace=True)

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())

for df in [X_train, X_test, submission]:
    df['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)
    df['Cabin_categorical'].fillna('Missing', inplace=True)

print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(submission.isnull().sum())

submission.Fare.fillna(X_train.Fare.median(), inplace=True)



def top_code(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])
 
for df in [X_train, X_test, submission]:
    df['Age'] = top_code(df, 'Age', 73)
    df['SibSp'] = top_code(df, 'SibSp', 4)
    df['Parch'] = top_code(df, 'Parch', 2)
    df['Family_size'] = top_code(df, 'Family_size', 7)




for var in ['Age',  'SibSp', 'Parch', 'Family_size']:
    print(var, ' max value: ', X_train[var].max())
for var in ['Age',  'SibSp', 'Parch', 'Family_size']:
    print(var, ' max value: ', submission[var].max())

X_train['Fare'], bins = pd.qcut(x=X_train['Fare'], q=8, retbins=True, precision=3)
X_test['Fare'] = pd.cut(x = X_test['Fare'], bins=bins, include_lowest=True)
submission['Fare'] = pd.cut(x = submission['Fare'], bins=bins, include_lowest=True)

submission.Fare.isnull().sum()
t1 = X_train.groupby(['Fare'])['Fare'].count() / np.float(len(X_train))
t2 = X_test.groupby(['Fare'])['Fare'].count() / np.float(len(X_test))
t3 = submission.groupby(['Fare'])['Fare'].count() / np.float(len(submission))
 
temp = pd.concat([t1,t2,t3], axis=1)
temp.columns = ['train', 'test', 'submission']
temp.plot.bar(figsize=(12,6))

X_train['Ticket_numerical'], bins = pd.qcut(x=X_train['Ticket_numerical'], q=8, retbins=True, precision=3)
X_test['Ticket_numerical'] = pd.cut(x = X_test['Ticket_numerical'], bins=bins, include_lowest=True)
submission['Ticket_numerical_temp'] = pd.cut(x = submission['Ticket_numerical'], bins=bins, include_lowest=True)
X_test.Ticket_numerical.isnull().sum()
submission.Ticket_numerical_temp.isnull().sum()
submission[submission.Ticket_numerical_temp.isnull()][['Ticket_numerical', 'Ticket_numerical_temp']]

submission.loc[submission.Ticket_numerical_temp.isnull(), 'Ticket_numerical_temp'] = X_train.Ticket_numerical.unique()[0]
submission.Ticket_numerical_temp.isnull().sum()

submission['Ticket_numerical'] = submission['Ticket_numerical_temp']
submission.drop(labels=['Ticket_numerical_temp'], inplace=True, axis=1)
submission.head()

for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()
def rare_imputation(variable, which='rare'):    
    
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    
    if which=='frequent':
        
        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], mode_label)
    
    else:
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')

rare_imputation('Cabin_categorical', 'frequent')
rare_imputation('Ticket_categorical', 'rare')

for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()
for var in categorical:
    print(var, submission[var].value_counts()/np.float(len(submission)))
    print()
    
for df in [X_train, X_test, submission]:
    df['Sex']  = pd.get_dummies(df.Sex, drop_first=True)
print(X_train.Sex.unique())
print(X_test.Sex.unique())
print(submission.Sex.unique())

def encode_categorical_variables(var, target):
        
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        submission[var] = submission[var].map(ordered_labels)
 

for var in categorical:
    encode_categorical_variables(var, 'Survived')
for df in [X_train, X_test, submission]:
    df.Fare = df.Fare.astype('O')
    df.Ticket_numerical = df.Ticket_numerical.astype('O')
for var in ['Fare', 'Ticket_numerical']:
    print(var)
    encode_categorical_variables(var, 'Survived')

print(X_train.head())

print(X_train.describe())

training_vars = [var for var in X_train.columns if var not in ['PassengerId', 'Survived']]

scaler = MinMaxScaler() 
scaler.fit(X_train[training_vars])

rf_model = RandomForestClassifier()
rf_model.fit(X_train[training_vars], y_train)
print(X_test.isnull().sum())
print(submission.isnull().sum()) 
pred = rf_model.predict_proba(X_train[training_vars])
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = rf_model.predict_proba(X_test[training_vars])
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

ada_model = AdaBoostClassifier()
ada_model.fit(X_train[training_vars], y_train)
 
pred = ada_model.predict_proba(X_train[training_vars])
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(X_test[training_vars])
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

logit_model = LogisticRegression()
logit_model.fit(scaler.transform(X_train[training_vars]), y_train)
 
pred = logit_model.predict_proba(scaler.transform(X_train[training_vars]))
print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(scaler.transform(X_test[training_vars]))
print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

pred_ls = []
for model in [rf_model, ada_model, logit_model]:
    pred_ls.append(pd.Series(model.predict_proba(X_test[training_vars])[:,1]))
print(pred_ls) 
final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
print(final_pred)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_pred)))









