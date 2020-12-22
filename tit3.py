import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


def data_preparation():
    titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])
    titanic_df = titanic_df.drop(['body', 'boat', 'cabin', 'home.dest', 'ticket'], axis=1)


    def get_status(x):
        for z in str(x).split():
            if z[-1] == '.':
                return z

    titanic_df['status'] = pd.DataFrame(titanic_df['name'].apply(get_status))


    median_status = titanic_df.groupby('status').median()["age"]

    median_status_list = [median_status[status] for status in titanic_df['status']]
    titanic_df['status_median'] = pd.DataFrame(median_status_list)
    titanic_df['age'] = titanic_df['age'].fillna(titanic_df['status_median'])
    titanic_df.drop(['status_median','name'], axis=1, inplace=True)
    nan_df = titanic_df[titanic_df.isna().any(axis=1)]
    mediana_pclass = titanic_df.groupby('pclass').median()["fare"]

    def get_median_pclass_list():
        median_pclass_list = [mediana_pclass[pclass] for pclass in titanic_df['pclass']]
        return median_pclass_list

    titanic_df['pclass_median_fare'] = pd.DataFrame(get_median_pclass_list())
    titanic_df['fare'] = titanic_df['fare'].fillna(titanic_df['pclass_median_fare'])
    titanic_df.drop(['pclass_median_fare'], axis=1, inplace=True)
    nan_df = titanic_df[titanic_df.isna().any(axis=1)]
    titanic_df['fam_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1
    usual_status = ['Miss.', 'Mlle.', 'Mme', 'Mr.', 'Mrs.', 'Ms.']
    status_state_list = ['usual_status' if str(x) in usual_status else 'special_status' for x in titanic_df['status']]
    titanic_df['status_state'] = pd.DataFrame(status_state_list)

    def division_into_age_groups(x):
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

    titanic_df['age_range'] = pd.DataFrame(titanic_df['age'].apply(division_into_age_groups))
    titanic_df['age_and_class.factor'] = titanic_df['age_range'] * (titanic_df['pclass']**2)
    titanic_df.drop(['sibsp', 'parch', 'status', 'age_range'], axis=1, inplace=True)

    titanic_l = titanic_df[['survived']]
    titanic_df = titanic_df.drop(['survived'], axis=1)
    titanic_num = titanic_df.drop(['embarked', 'sex', 'status_state'], axis=1)
    titanic_cat = titanic_df[['embarked', 'sex', 'status_state']]
    imputer_num = SimpleImputer(strategy='median')
    imputer_num.fit(titanic_num)
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_cat.fit(titanic_cat)

    titanic_pre_tr_num = imputer_num.transform(titanic_num)
    titanic_tr_num = pd.DataFrame(titanic_pre_tr_num, columns=titanic_num.columns, index=titanic_num.index)
    titanic_pre_tr_cat = imputer_cat.transform(titanic_cat)
    titanic_tr_cat = pd.DataFrame(titanic_pre_tr_cat, columns=titanic_cat.columns, index=titanic_cat.index)
    cat_encoder = OrdinalEncoder()           #OneHotEncoder()
    titanic_pre_tr_cat_ord = cat_encoder.fit_transform(titanic_tr_cat)

    titanic_tr_cat_ord = pd.DataFrame(titanic_pre_tr_cat_ord, columns=titanic_cat.columns, index=titanic_cat.index)
    # std_scaler = StandardScaler()
    # titanic_pre_tr_num_sc = std_scaler.fit_transform(titanic_tr_num)
    mm_scaler = MinMaxScaler()
    titanic_pre_tr_num_sc = mm_scaler.fit_transform(titanic_tr_num)

    titanic_tr_num_sc = pd.DataFrame(titanic_pre_tr_num_sc, columns=titanic_num.columns, index=titanic_num.index)
    titanic_prepreprepard = titanic_tr_cat_ord.join(titanic_tr_num_sc)
    titanic_preprepard = titanic_prepreprepard.join(titanic_l)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in split.split(titanic_preprepard, titanic_preprepard['sex']):
        strat_tit_train_set = titanic_preprepard.reindex(index= train_index)
        strat_tit_test_set = titanic_preprepard.reindex(index= test_index)

    titanic_prepard = strat_tit_train_set.drop('survived', axis=1)
    titanic_prepard_labels = strat_tit_train_set['survived'].copy()
    le = LabelEncoder()
    le.fit(titanic_prepard_labels)
    return titanic_prepard, titanic_prepard_labels

def data_training(t,t_l):

    forest_reg = RandomForestClassifier()
    forest_reg.fit(t, t_l)
    # titanic_pred = forest_reg.predict(titanic)
    #scores = cross_val_score(forest_reg, titanic, titanic_labels, scoring="accuracy", cv=3)
    # frc_mse = mean_squared_error(titanic_labels, titanic_pred)
    # frc_rmse_scores = np.sqrt(-scores)
    # print(frc_rmse_scores)
    # acc = accuracy_score(titanic_labels, titanic_pred)
    #print(scores)
    importances = forest_reg.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_reg.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(t.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])+ " " + t.columns[indices[f]])

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(t.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    degrees = 20
    plt.xticks(rotation=degrees, fontsize=8)
    plt.xticks(range(t.shape[1]), t.columns[indices])
    plt.xlim([-1, t.shape[1]])
    plt.show()

if __name__ == "__main__":
    titanic, titanic_labels = data_preparation()
    data_training(titanic, titanic_labels)