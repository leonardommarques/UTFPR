# -------------------------------------------- #
# -- Simulation of data sets -- #
# -------------------------------------------- #

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import time



import numpy as np
from functools import reduce

folder_ = "/Users/leo/Documents/Estudos/UTFPR/reconehcimento_de_padroes/trabalho_final/"

make_single_best = True
# ---------------------------------------------- #
# -- parallel function -- #
# ---------------------------------------------- #

from multiprocessing import Pool

def parallelize_dataframe(df, func, num_partitions = 10, n_jobs=8):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# ---------------------------------------------- #


dsc = DSC(clf_list, k=15)

# dsc.classifiers_train_val
# dsc.classifiers



# -- auxiliary datafarmes -- #
# dsc
acc_df = pd.DataFrame({
    'n_samples_': [0]
    , 'n_features_': [0]
    , 'n_informative_': [0]
    , 'n_classes_': [0]
    , 'scale_': [0]
    , 'k': [0]
})

# single best
acc_single_best_df = acc_df.copy().drop('k', 1)

np.random.seed(123)
simulation_index = 0
while simulation_index < 1:
    simulation_index = simulation_index + 1

    print('rodada: {}'.format(simulation_index ))

    make_data_error = True
    n_errors = 0
    while make_data_error:

        try:
            n_samples_ = np.random.randint(500, 2000, 1)[0]
            n_features_ = np.random.randint(2, 100, 1)[0]
            n_informative_ = np.random.randint(2, n_features_, 1)[0]

            if n_informative_ == 2:
                n_classes_ = 2
            # elif n_classes_ > 2.0 ** n_informative_:
            #     n_classes_ = n_classes_ // 2
            else:
                n_classes_ = np.random.randint(2, n_informative_, 1)[0]

            scale_ = np.random.uniform(.5, 2.5)

            X, y = make_classification(
                n_samples=n_samples_
                , n_features=n_features_
                , n_redundant=0
                , n_informative=n_informative_
                , n_classes = n_classes_
                , scale = scale_
                , n_clusters_per_class=1
                , random_state = simulation_index
            )
            make_data_error = False
        except:
            print('data make error')
            n_errors = n_errors + 1
            simulation_index = simulation_index + 1

        if n_errors >= 5:
            nreak


    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9)

    dsc.fit(X_train, y_train)


    aux_acc_df = pd.DataFrame({
          'n_samples_': [n_samples_]
        , 'n_features_': [n_features_]
        , 'n_informative_': [n_informative_]
        , 'n_classes_': [n_classes_]
        , 'scale_': [scale_]
        , 'k': 0
        , 'sim_id': simulation_index
    })


    k_range = list(range(0,51, 5))
    k_range[0] = 3
    for i in range(max(k_range)+10,101, 10):
        k_range.append(i)
    for i in range(max(k_range)+100,301, 100):
        k_range.append(i)

    strategy = 'ds_la_ola'
    # strategy = 'ds_la_lca'
    # strategy = 'ds_knora_eliminate'

    if strategy == 'ds_la_ola':
        make_single_best = True
    else:
        make_single_best = False

    start = time.time()

    # for k_ in k_range:
    #     dsc.k = k_
    #
    #     def pred_func(xxx):
    #         result = dsc.predict(X_train, y_train, xxx, strategy=strategy)
    #         return result
    #
    #     aux = parallelize_dataframe(X_test, pred_func, num_partitions=10, n_jobs=16)
    #     aux_acc = accuracy_score(y_test, aux['DSC_pred']).round(4)
    #
    #     print('rodada: {}'.format(simulation_index))
    #     print('K: {}'.format(k_))
    #     print('Acc:', aux_acc)
    #     print('\n')
    #
    #     aux_acc_df['k'] = dsc.k
    #     aux_acc_df['accuracy'] = aux_acc
    #     aux_acc_df['strategy'] = strategy
    #
    #     acc_df = pd.concat([acc_df, aux_acc_df])

    # --------------------- #
    # -- acc single best -- #
    # --------------------- #

    if make_single_best:
        single_best_estimator = dsc.accuracy(X_train, y_train).T.sort_values(by='accuracy').tail(1).T.columns[0]
        single_best_scc = dsc.accuracy(X_test, y_test)[[single_best_estimator]]
        # single_best_scc = single_best_scc.T.sort_values(by='accuracy').tail(1)
        single_best_scc = single_best_scc.iloc[0,0]

        aux_acc_single_best_df = aux_acc_df.copy().drop('k', 1)
        aux_acc_single_best_df['accuracy'] = single_best_scc
        aux_acc_single_best_df['strategy'] = 'single_best'
        acc_single_best_df = pd.concat([acc_single_best_df, aux_acc_single_best_df])

        if simulation_index % 10 == 0:
            acc_single_best_df.to_csv(folder_ + 'acc_single_best_df_' + 'rodada_' + str(simulation_index) + 'csv')

    # -- save every 10 -- #
    if simulation_index % 10 == 0:
        acc_df.to_csv(folder_ + strategy + '_acc_df_' + 'rodada_' + str(simulation_index) + 'csv')



end = time.time()

(end - start)/60
# ds_knora_eliminate: 64 minutos
# lca: 6 minutos
# OLA: 2 minutos

acc_df_old = acc_df.copy()
acc_single_best_df_old = acc_single_best_df.copy()

# -- remove first line -- #
acc_df = acc_df.dropna(subset=['accuracy'])
acc_single_best_df = acc_single_best_df.dropna(subset=['accuracy'])

acc_df['sim_id'].unique()
acc_df.head()

for i in acc_df['sim_id'].unique():
    index = acc_df['sim_id'] == i
    plt.plot(acc_df[index]['k'], acc_df[index]['accuracy'], label=i)


acc_df['k']
