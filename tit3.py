import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])
titanic_df = titanic_df.drop(['body', 'boat', 'cabin', 'home.dest', 'ticket'], axis=1)
def a(x):
    for z in str(x).split():
        if z[-1] == '.':
            return z
titanic_df['status'] = pd.DataFrame(titanic_df['name'].apply(a))
nan_df = titanic_df[titanic_df.isna().any(axis=1)]
# print(titanic_df.groupby('status').median()["age"])
l = titanic_df.groupby('status').median()["age"]
k =[l[i] for i in titanic_df['status']]
titanic_df['status_median'] = pd.DataFrame(k)
titanic_df['age'] = titanic_df['age'].fillna(titanic_df['status_median'])
titanic_df.drop(['status_median','name'], axis=1, inplace=True)
nan_df = titanic_df[titanic_df.isna().any(axis=1)]
f = titanic_df.groupby('pclass').median()["fare"]
j =[f[i] for i in titanic_df['pclass']]
titanic_df['pclass_median_fare'] = pd.DataFrame(j)
titanic_df['fare'] = titanic_df['fare'].fillna(titanic_df['pclass_median_fare'])
titanic_df.drop(['pclass_median_fare'], axis=1, inplace=True)
nan_df = titanic_df[titanic_df.isna().any(axis=1)]
titanic_df['fam_state'] = titanic_df['sibsp'] + titanic_df['parch'] + 1
usual_status = ['Miss.', 'Mlle.', 'Mme', 'Mr.', 'Mrs.', 'Ms.']
g = ['usual_status' if str(x) in usual_status else 'special_status' for x in titanic_df['status']]
titanic_df['status_state'] = pd.DataFrame(g)
def b(x):
    y = int(x)
    if y < 6:
        return 1
    elif y < 12:
        return 2
    elif y < 18:
        return 3
    elif y < 65:
        return 4
    else:
        return 5
titanic_df['age_range'] = pd.DataFrame(titanic_df['age'].apply(b))
titanic_df['age.class.factor'] = titanic_df['age_range'] * (titanic_df['pclass']**2)
#print(titanic_df['age.class.factor'].unique())
titanic_df.drop(['sibsp', 'parch', 'status', 'age_range'], axis=1, inplace=True)
# print(nan_df.head())
# print(titanic_df.head())
# titanic_df.hist(bins=50,figsize=(20,15))
# plt.show()
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
# for train_index, test_index in split.split(titanic_df, titanic_df['sex']):
#     strat_tit_train_set = titanic_df.reindex(index= train_index)
#     strat_tit_test_set = titanic_df.reindex(index= test_index)

# titanic = strat_tit_train_set.drop('survived', axis=1)
# titanic_labels = strat_tit_train_set['survived'].copy()
titanic_l= titanic_df[['survived']]
titanic_df = titanic_df.drop(['survived'], axis=1)
titanic_num = titanic_df.drop(['embarked', 'sex', 'status_state'], axis=1)
titanic_cat = titanic_df[['embarked', 'sex', 'status_state']]
imputer_num = SimpleImputer(strategy='median')
imputer_num.fit(titanic_num)
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_cat.fit(titanic_cat)
# print(imputer_cat.statistics_)
# print(titanic_num.median().values)
X = imputer_num.transform(titanic_num)
titanic_tr_num = pd.DataFrame(X, columns=titanic_num.columns, index=titanic_num.index)
Y = imputer_cat.transform(titanic_cat)
titanic_tr_cat = pd.DataFrame(Y, columns=titanic_cat.columns, index=titanic_cat.index)
cat_encoder = OrdinalEncoder()           #OneHotEncoder()
W = cat_encoder.fit_transform(titanic_tr_cat)
#print(type(W))
titanic_tr_cat_ord = pd.DataFrame(W, columns=titanic_cat.columns, index=titanic_cat.index)
std_scaler = StandardScaler()
Z = std_scaler.fit_transform(titanic_tr_num)
# print(type(Z))
titanic_tr_num_sc = pd.DataFrame(Z, columns=titanic_num.columns, index=titanic_num.index)
titanic_prepreprepard = titanic_tr_cat_ord.join(titanic_tr_num_sc)
titanic_preprepard = titanic_prepreprepard.join(titanic_l)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(titanic_preprepard, titanic_preprepard['sex']):
    strat_tit_train_set = titanic_preprepard.reindex(index= train_index)
    strat_tit_test_set = titanic_preprepard.reindex(index= test_index)

titanic = strat_tit_train_set.drop('survived', axis=1)
titanic_labels = strat_tit_train_set['survived'].copy()
le = LabelEncoder()
le.fit(titanic_labels)
#print(titanic_labels.head())
#titanic_prepard= cat_encoder.fit_transform(titanic_tr_cat)
# tp = titanic_prepard.values.reshape(-1, 1)
# tl = titanic_labels.values.reshape(-1, 1)
#print(titanic_prepard)
# num_pipline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler()),])
# cat_pipline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('one_hot', OneHotEncoder),])
# num_pipline = Pipeline([('std_scaler', StandardScaler()),])
# cat_pipline = Pipeline([('one_hot', OneHotEncoder),])
# num_attribs =list(titanic_tr_num)
# cat_attribs =list(titanic_tr_cat)
# full_pipeline = ColumnTransformer([('num', num_pipline, num_attribs), ('cat', cat_pipline, cat_attribs ),])
# titanic_prepared = full_pipeline.fit_transform(titanic)
forest_reg = RandomForestClassifier()
# forest_reg.fit(titanic, titanic_labels)
# titanic_pred = forest_reg.predict(titanic)
scores = cross_val_score(forest_reg, titanic, titanic_labels, scoring="accuracy", cv=3)
# frc_mse = mean_squared_error(titanic_labels, titanic_pred)
# frc_rmse_scores = np.sqrt(-scores)
# print(frc_rmse_scores)
# acc = accuracy_score(titanic_labels, titanic_pred)
print(scores)