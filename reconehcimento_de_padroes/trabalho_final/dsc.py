# -------------------------------------------- #
# -- Dynamic Selection fo Classifiers (DSC) -- #
# -------------------------------------------- #

# -- import -- #

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import copy
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100)



# ---------------------------------------- #
# -- aux functions
# ---------------------------------------- #

def gen_gaussian(mu = [0, 0], sigma = [1, 0, 0, 1], n=400):


    sigma = np.array(sigma).reshape([2, 2])
    sample = np.random.multivariate_normal(mu, sigma, n)
    sample = np.array(sample)

    sample = pd.DataFrame(sample)
    sample.columns = ['x1', 'x2']


    return sample


# -- plot
def scatterplot(aux_da):

    for i in aux_da['class'].unique():
        index = aux_da['class'] == i
        plt.scatter(aux_da[index]['x1'], aux_da[index]['x2'], label=i, s=7)

    plt.legend(loc='upper left')


# ------------------- #
# -- DSC estimator -- #
# ------------------- #

import multiprocessing
from sklearn.base import BaseEstimator


class DSC(BaseEstimator):
    def __init__(self, classifiers, k=15):

        self.classifiers = classifiers
        self.classifiers_train_val = copy.deepcopy(classifiers)
        self.k = k

    def fit(self, X, y):

        from sklearn.metrics import accuracy_score
        import pandas as pd

        if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
            X = X.copy()


        self.X = X
        self.y = y

        # -- Fit -- #
        print('fitting train classifiers...')
        for label_, clf_ in self.classifiers.items():
            clf_.fit(X, y)

        # -- accuracy -- #
        train_acc = {label_: accuracy_score(y, clf_.predict(X)) for label_, clf_ in self.classifiers.items()}
        train_acc = pd.DataFrame([train_acc], index=['train_accuracy'])
        self.train_accuracy = train_acc


        # -- train validation -- #
        print('splitting train validation')
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, train_size=.9)

        self.X_train_ = X_train_
        self.X_val_ = X_val_
        self.y_train_ = y_train_
        self.y_val_ = y_val_

        # classifiers trained with train validation
        print('fitting train validation classifiers')
        for label_, clf_ in self.classifiers_train_val.items():
            clf_.fit(self.X_train_, self.y_train_)


    def predict(self, X_train_, y_train_, X_pred_, k=0, strategy='ds_la_ola'):

        # strategy: ds_la_ola, ds_la_lca

        from sklearn.metrics import accuracy_score
        from scipy.spatial import distance
        import numpy as np
        import pandas as pd

        X_train_ = X_train_.copy()
        y_train_ = y_train_.copy()
        X_pred_ = X_pred_.copy()

        # -- crete data frame to assert the indexes -- #
        df_ = pd.DataFrame(X_train_)
        df_['y'] = y_train_

        X_train_ = df_[[i for i in df_.columns if i not in ['y', 'dist']]]

        if k == 0:
            k = self.k

        if strategy == 'ds_la_ola':
            predict_dsc = self.predict_ds_la_ola

        elif strategy == 'ds_la_lca':
            predict_dsc = self.predict_ds_la_lca

        elif strategy == 'ds_knora_eliminate':
            predict_dsc = self.predict_knora_eliminiate


        dsc_pred = X_pred_.apply(lambda x: predict_dsc(x.values, X_train_, y_train_, k=k, clf_list_=self.classifiers), 1)

        DSC_pred = [i['DSC_pred'] for i in dsc_pred]
        selected_clf = [i['selected_clf'] for i in dsc_pred]

        result_df = pd.DataFrame()
        result_df['DSC_pred'] = DSC_pred
        result_df['selected_clf'] = selected_clf

        # print(type(dsc_pred['final_k']))
        # print('final_k exists: ' + str('final_k' in dsc_pred[0]))
        #
        # if 'final_k' in dsc_pred[0]:
        #     final_k = [i['final_k'] for i in dsc_pred]
        #     result_df['final_k'] = final_k

        # -- result -- #
        result = result_df

        return result

    def accuracy(self, X_,y_true):

        train_acc = {label_: accuracy_score(y_true, clf_.predict(X_)) for label_, clf_ in self.classifiers.items()}
        train_acc = pd.DataFrame([train_acc], index=['accuracy'])

        return train_acc

    # ---------------------------- #
    # -- prediction strategies --  #
    # ---------------------------- #

    # -- Algorithm 2 -- #
    def predict_ds_la_ola(self, X_pred_, X_train_, y_train_, k, clf_list_):

        X_pred_ = X_pred_.reshape(1, X_train_.shape[1])
        X_train_ = X_train_.copy()

        # ------------- #
        # -- Get KNN -- #
        # ------------- #
        X_train_['dist'] = [distance.euclidean(vec, X_pred_) for vec in X_train_.values]
        X_train_ = X_train_.sort_values(by='dist', ascending=True)
        X_train_ = X_train_.drop('dist', 1)

        X_knn = X_train_.head(k).copy()
        y_knn = y_train_.loc[X_knn.index].copy()

        knn_preds = {
            label_: clf_.predict(X_knn) for label_, clf_ in clf_list_.items()
        }

        # check for classifications unanimity
        if len(np.unique(pd.DataFrame(knn_preds).values.reshape(1, -1))) == 1:
            unanimity = True
        else:
            unanimity = False

        knn_acc = {label_: accuracy_score(y_knn, pred_) for label_, pred_ in knn_preds.items()}
        knn_acc = pd.DataFrame([knn_acc], index=['accuracy']).T
        knn_acc = knn_acc.sort_values(by='accuracy', ascending=False)

        # ------------------------- #
        # -- prediction -- #
        # ------------------------- #
        selected_clf_index = knn_acc.head(1).index[0]
        selected_clf = clf_list_[selected_clf_index]
        DSC_pred = selected_clf.predict(X_pred_)
        DSC_pred = DSC_pred[0]

        # -- results -- #
        results = {
            # 'knn_acc': knn_acc
            'DSC_pred': DSC_pred
            , 'selected_clf': selected_clf_index
        }

        return results

    def predict_ds_la_lca(self, X_pred_, X_train_, y_train_, k, clf_list_):

        X_pred_ = X_pred_.reshape(1, X_train_.shape[1])
        X_train_ = X_train_.copy()

        knn_accuracy_list = {}

        for label_, clf_ in clf_list_.items():
            y_pred_ = clf_.predict(X_pred_)
            index = y_train_ == y_pred_

            X_train_filtered = X_train_.loc[index.index,]

            # -- Get KNN -- #
            X_train_filtered['dist'] = [distance.euclidean(vec, X_pred_) for vec in X_train_filtered.values]

            X_knn_ = X_train_filtered.sort_values(by='dist', ascending=True).copy()
            X_knn_ = X_knn_.drop('dist', 1)
            X_knn_ = X_knn_.head(k)

            y_knn = y_train_.loc[X_knn_.index].copy()

            y_knn_pred = clf_.predict(X_knn_)

            knn_accuracy_list[label_] = accuracy_score(y_knn, y_knn_pred)

            # -- Get best classifier -- #
            knn_accuracy_df = pd.DataFrame(knn_accuracy_list, index=['accuracy'])
            knn_accuracy_df = knn_accuracy_df.T.sort_values(by='accuracy')
            knn_accuracy_df = knn_accuracy_df.tail(1)

            selected_clf_index = knn_accuracy_df.index.values[0]

        y_test_pediction = clf_list_[selected_clf_index].predict(X_pred_)
        y_test_pediction = y_test_pediction[0]

        # -- results -- #
        results = {
            # 'knn_acc': knn_acc
            'DSC_pred': y_test_pediction
            , 'selected_clf': selected_clf_index
        }

        return results

    def predict_knora_eliminiate(self, X_pred_, X_train_, y_train_, k, clf_list_):
        k_aux = k

        # print(type(X_pred_))

        # X_pred_ = X_pred_.reshape(1, X_train_.shape[1])

        X_train_ = self.X_train_.copy()
        X_val_ = self.X_val_.copy()
        y_train_ = self.y_train_.copy()
        y_val_ = self.y_val_.copy()

        X_val_['dist'] = [distance.euclidean(vec, X_pred_) for vec in X_val_.values]

        X_val_ = X_val_.sort_values(by='dist', ascending=True)
        X_val_ = X_val_.drop('dist', 1)
        y_val_ = y_val_.loc[X_val_.index].copy()

        # y_val_ = y_val_ + 1000
        k_aux_was_never_zero = True
        while k_aux >= 0:
            # print(k_aux)

            # -- knn -- #
            X_knn = X_val_.head(k_aux).copy()
            y_knn = y_val_.loc[X_knn.index].copy()

            aux_knnpreds = {}
            for label_, clf_ in self.classifiers_train_val.items():
                # clf_.fit(X_train_, y_train_)
                clf_pred_ = clf_.predict(X_knn)
                # clf_pred_ = clf_pred_ == np.array(y_knn)
                aux_df = pd.DataFrame({'pred': clf_pred_}, index=y_knn.index)
                aux_df['y'] = y_knn
                idx = aux_df['y'] == aux_df['pred']
                aux_df = aux_df[idx]
                correct_y_knnpred_idx = aux_df.index

                aux_knnpreds[label_] = correct_y_knnpred_idx

            aux_knnpreds_df = {}
            for label_, clf_ in aux_knnpreds.items():
                aux_df = pd.DataFrame({label_: list(clf_)}, index=clf_)
                aux_knnpreds_df[label_] = aux_df

            aux_knnpreds_df = reduce((lambda x, y: x.join(y, how='outer')), aux_knnpreds_df.values())
            if aux_knnpreds_df.shape[0] > 0:
                ensenble_ = aux_knnpreds_df.dropna(axis='columns')
                ensenble_ = list(ensenble_.columns)
            else:
                ensenble_ = list()

            if k_aux_was_never_zero:
                if k_aux > 1:
                    if len(ensenble_) <= 0:
                        k_aux = k_aux - 1
                    else:
                        break
                elif k_aux <= 1:
                    k_aux = k
                    k_aux_was_never_zero = False
                    print('k became Zero')
            else:
                ensenble_ = aux_knnpreds_df.isna().sum()
                ensenble_ = ensenble_[ensenble_ == ensenble_.min()]
                ensenble_ = list(ensenble_.index)
                break

        ensenble_list = {label_: clf_ for label_, clf_ in clf_list_.items() if label_ in ensenble_}
        ensemble_pred = [clf_.predict(X_pred_.reshape(1, X_train_.shape[1])) for clf_ in ensenble_list.values()]

        selected_clf_index = ensenble_
        DSC_pred = int(np.mean(ensemble_pred))

        self.k = k_aux

        # print('final value pf k: ' + str(k_aux))
        # print('final value pf self.k: ' + str(self.k))
        # print('sua mãe')

        results = {
            # 'knn_acc': knn_acc
            'DSC_pred': DSC_pred
            , 'selected_clf': selected_clf_index
            , 'final_k': k_aux
        }

        return results


# ---------------------------------------- #
# -- data - #
# ---------------------------------------- #
aux = gen_gaussian([-1, 1], [2,1,1,2])
aux['class'] = '0'

aux1 = gen_gaussian([1, -1], [3,.5,.5,2])
aux1['class'] = '1'

da= pd.concat([aux, aux1])
da = da.reset_index()

# scatterplot(da)

features = ['x1', 'x2']
target = ['class']

X_train, X_test, y_train, y_test = train_test_split(da[features], da[target], train_size=.8)

# -------------------------- #
# -- fit --  #
# -------------------------- #

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score


# -- classifiers -- #
clf_list = {
    'svc': SVC()
    , 'forest': RandomForestClassifier(n_estimators=3)
    , 'lreg': LogisticRegression()
    , 'nn': MLPClassifier()
}














for label_, clf_ in clf_list.items():
    print(label_)
    clf_.fit(X_train, y_train)

# -- accuracy -- #
train_acc = {label_: accuracy_score(y_train, clf_.predict(X_train)) for label_, clf_ in clf_list.items()}
train_acc = pd.DataFrame([train_acc], index = ['train'])

test_acc = {label_: accuracy_score(y_test, clf_.predict(X_test)) for label_, clf_ in clf_list.items()}
test_acc = pd.DataFrame([test_acc], index = ['test'])

acc_df = pd.concat([train_acc, test_acc]).T

# --------------------------------- #
# -- DSC -- #
# --------------------------------- #

dsc = DSC(clf_list, k=15)
dsc.fit(X_train, y_train)

aux = dsc.predict(X_train, y_train, X_test)
aux.head()
accuracy_score(y_test, aux['DSC_pred'])
acc_df

# --------------------------------- #
# -- Another dataset -- #
# --------------------------------- #

def gen_gaussian_mix(mu, sigma, n):

    index = len(mu)
    index = np.random.choice(index, size=n)

    mu = [mu[0], mu[1]]
    sigma = [np.array(sigma[0]).reshape([2, 2]), np.array(sigma[1]).reshape([2, 2])]
    sample = [np.random.multivariate_normal(mu[i], sigma[i]) for i in index]
    sample = np.array(sample)

    sample = pd.DataFrame(sample)
    sample.columns = ['x1', 'x2']


    return sample


S11 = [.2,0,0,2]
S12 = [3,0,0,.5]
S21 = [5,0,0,.5]
S22 = [7,0,0,.5]
S3 = [8,0,0,.5]

mu11 = [0,3]
mu12 = [11, -2]
mu21 = [3, -2]
mu22 = [7.5, 4]
mu3 = [7,2]

aux = gen_gaussian_mix([mu11, mu12], [S11, S12], 500)
aux['class'] = 0
aux1 = gen_gaussian_mix([mu21, mu22], [S21, S22], 500)
aux1['class'] = 1
da = pd.concat([aux, aux1])
da = da.reset_index()

scatterplot(da)

X_train, X_test, y_train, y_test = train_test_split(da[features], da[target], train_size=.8)

dsc = DSC(clf_list, k=15)
dsc.fit(X_train, y_train)

train_acc = dsc.accuracy(X_=X_train, y_true=y_train)
train_acc.index = ['train_accuracy']
test_acc = dsc.accuracy(X_=X_test, y_true=y_test)
test_acc.index = ['test_accuracy']
acc_df = pd.concat([train_acc, test_acc]).T

dsc_preds = dsc.predict(X_train, y_train, X_test)
accuracy_score(y_test, dsc_preds['DSC_pred'])

# ----------------------------------- #
# -- another data set -- #
# ----------------------------------- #

np.random.seed(123)

def make_data(n=2000, seed=0):
    np.random.seed(seed)
    x1 = np.random.uniform(-5, 5, n)
    x2 = np.random.uniform(-5, 5, n)
    y = 0.05 * (3*x1 ** 3 + 10*x1 ** 2 -10*x1 + 1) > x2

    aux_df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})
    aux_df['class'] = aux_df['class'].apply(int)

    return aux_df

da=make_data()
da.head()

# scatterplot(da)

X_train, X_test, y_train, y_test = train_test_split(da[features], da[target], train_size=.8)


# --------------------------------- #
# -- DSC -- #
# --------------------------------- #

dsc = DSC(clf_list, k=15)
dsc.fit(X_train, y_train)

aux = dsc.predict(X_train, y_train, X_test)
aux.head()
np.unique(aux['selected_clf'])

train_acc = dsc.accuracy(X_=X_train, y_true=y_train)
train_acc.index = ['train_accuracy']
test_acc = dsc.accuracy(X_=X_test, y_true=y_test)
test_acc.index = ['test_accuracy']
acc_df = pd.concat([train_acc, test_acc]).T.sort_values(by='test_accuracy', ascending=False)


accuracy_score(y_test, aux['DSC_pred'])
acc_df
