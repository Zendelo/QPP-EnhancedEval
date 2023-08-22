import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, log_loss, SCORERS
import random
import matplotlib.pyplot as plt
import lightgbm as lgb
from syct import timer

from qpptk import plot_roc


class NestedCrossVal:
    """
    Implements the nested cross validation methodology.
    First the data is split in an outer loop to k folds, where each fold serves as a test set single time while all the
    other folds are used as train data. The test set is used for the final evaluation of the model.
    Inside the first loop, each train set is split into train validation (a.k.a dev) sets. In the inner loop the
    validation set is used for parameter optimization, i.e. to find the best parameters for the model. After the best
    parameters are found the model is then trained over all the train set (including the validation) and evaluated on
    the test set.
    The data is over the topics to avoid data leakage, the cost is that it may cause imbalanced data.
    (over stratified sampling)
    """

    def __init__(self, data_df):
        self.full_data = data_df

    @timer
    def outer_evaluation(self, run_path=''):
        topics = self.full_data['topic'].unique()
        test_predictions = []
        params_res = []
        features_imp = []
        kf = KFold(n_splits=10, random_state=321)
        for train_index, test_index in kf.split(topics):
            train_df = self.full_data.loc[self.full_data['topic'].isin(topics[train_index])]
            test_df = self.full_data.loc[self.full_data['topic'].isin(topics[test_index])]
            print(f'Train:')
            print_data_summary(train_df)
            print(f'Test:')
            print_data_summary(test_df)
            _predictions, params_df, features_import = self.find_best_params(train_df, test_df)
            test_predictions.append(pd.Series(index=test_df.index, data=_predictions))
            params_res.append(params_df)
            features_imp.append(pd.DataFrame(features_import, index=train_df.drop(['label', 'topic'], axis=1).columns))
        full_test = pd.concat(test_predictions)
        full_test.to_pickle(f'full_prediction_set.pkl')
        full_test.to_csv(f"{run_path}_LGBM.pre", sep=' ', index=True, header=False, float_format="%.4f")
        df = pd.concat(params_res)
        df.to_pickle('full_cv_result.pkl')
        features_df = pd.concat(features_imp).set_index('features')
        features_df = features_df.groupby('features').mean()
        features_df.to_pickle('features_importance.pkl')
        print('The AucRoc after CV is:', roc_auc_score(self.full_data['label'], full_test))
        plot_roc(self.full_data['label'], full_test, 'LGBM-Predictor')
        return roc_auc_score(self.full_data['label'], full_test)

    @timer
    def find_best_params(self, train_df, test_df):
        estimator = lgb.LGBMClassifier()
        param_grid = [
            # {'objective': ['binary'], 'metric': ['auc'], 'boosting_type': ['dart'], 'is_unbalance': [True],
            #  'num_leaves': [5], 'min_data_in_leaf': [5], 'lambda_l1': [0.7], 'lambda_l2': [0.7],
            #  'feature_fraction': [0.0625], 'bagging_fraction': [0.25, 0.5, 0.75], 'bagging_freq': [1, 5, 10]}
            {'objective': ['binary'], 'boosting_type': ['dart'],
             'metric': ['auc'],
             'bagging_fraction': [0.25, 0.5, 0.75],
             'num_leaves': [5],
             'learning_rate': [0.05, 0.1],
             'feature_fraction': [0.0625, 0.125, 0.25],
             # 'feature_fraction': [0.0625, 0.125],
             'bagging_freq': [1],
             'lambda_l1': [0.75],  # helps in feature selection - minimizes to median of data
             'lambda_l2': [0.75],  # minimizes the mean of data
             'is_unbalance': [True],
             'num_iterations': [50, 100, 200, 300],
             'min_data_in_leaf': [5], 'verbose': [-1]},
            # {'objective': ['binary'], 'boosting_type': ['dart'],
            #  'metric': ['auc'],
            #  'num_leaves': [5],
            #  'learning_rate': [0.01, 0.05, 0.1],
            #  'feature_fraction': [0.0625, 0.125, 0.25],
            #  # 'feature_fraction': [0.0625, 0.125],
            #  'bagging_freq': [0],
            #  'lambda_l1': [0.75],  # helps in feature selection - minimizes to median of data
            #  'lambda_l2': [0.75],  # minimizes the mean of data
            #  'is_unbalance': [True],
            #  'num_iterations': [100, 150, 200],
            #  'min_data_in_leaf': [5], 'verbose': [-1]}
        ]
        # score function should be specified here (AUC)
        # cv = None - default value 5-folds
        clf = GridSearchCV(estimator, param_grid, n_jobs=40, refit=True, cv=3, verbose=1, error_score='raise',
                           scoring='roc_auc', return_train_score=True)
        fim = clf.fit(train_df.drop(['label', 'topic'], axis=1), train_df['label'])
        _df = pd.DataFrame(fim.cv_results_)
        df = _df.sort_values('mean_test_score', ascending=False).head(10)
        param_cols = df.columns[df.columns.str.startswith('param_')].tolist()
        # df.loc[:, param_cols + ['mean_test_score']].to_pickle('cv_results.pkl')
        # predictions = fim.predict_proba(test_df.drop(['topic', 'label'], axis=1))[:, 1]
        # print('CV model AUC of ROC of prediction is:', roc_auc_score(test_df['label'], predictions))
        # print(f'Logloss = {log_loss(test_df["label"], predictions): 0.2f}')
        # print(fim.best_estimator_)
        # print(fim.best_params_)
        # print(fim.best_score_)
        _best_params = fim.best_params_
        _best_params.update(verbose=4, num_threads=10)
        # _best_params.update(num_threads=10)
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(train_df.drop(['label', 'topic'], axis=1), train_df['label'])
        gbm = lgb.train(_best_params, lgb_train)
        # print('Starting predicting...')
        # predict
        predictions = gbm.predict(test_df.drop(['topic', 'label'], axis=1))
        # assert (fim.best_estimator_.feature_importances_ == gbm.feature_importance('split')).all(), \
        #     f'{fim.best_estimator_.feature_importances_}\n {gbm.feature_importance("split")}'
        # eval
        # print('The AUC of ROC of prediction is:', roc_auc_score(test_df['label'], y_pred))
        # try:
        #     print('The type of best_est ', type(best_est))
        #     # save model to file
        #     gbm.save_model('model.txt')
        # except AttributeError as er:
        #     print(er)
        # lgb.plot_importance(gbm)
        # lgb.plot_split_value_histogram(gbm, 'max-var')
        # plt.show()
        # exit()
        return predictions, df.loc[:, param_cols + ['mean_test_score', 'mean_train_score']], {
            'features': gbm.feature_name(), 'split': gbm.feature_importance('split'),
            'gain': gbm.feature_importance('gain')}


def print_best_params(n=15):
    cv_res = 'full_cv_result.pkl'
    _df = pd.read_pickle(cv_res)
    df = _df.sort_values('mean_test_score', ascending=False).head(n)
    for col in df.columns:
        print(col)
        if col.endswith('score'):
            continue
        else:
            print(df.groupby(col).count())
    else:
        print(df[['mean_test_score', 'mean_train_score']])
    print(f'max score: {df["mean_test_score"].max()}')


def print_data_summary(data_df=None):
    data_df = pd.read_pickle('data/data_df.pkl') if data_df is None else data_df
    print(f"Number of positive samples: {data_df['label'].sum()}")
    print(f"Ratio of positive samples: {data_df['label'].sum() / len(data_df)}")


def plot_feature_importance():
    features_df = pd.read_pickle('features_importance.pkl')

    by_split = features_df['split'].sort_values()
    fig, ax = plt.subplots()
    ylocs = np.arange(len(by_split))
    ax.barh(ylocs, by_split, align='center')
    ax.set_yticks(ylocs)
    ax.set_yticklabels(by_split.index)
    ax.set_title('Feature importance by split')
    # plt.show()
    fig.savefig('feat_split_imp_ndcg_03.eps')

    by_gain = features_df['gain'].sort_values()
    fig, ax = plt.subplots()
    ylocs = np.arange(len(by_gain))
    ax.barh(ylocs, by_gain, align='center')
    ax.set_yticks(ylocs)
    ax.set_yticklabels(by_gain.index)
    ax.set_title('Feature importance by gain')
    # plt.show()
    fig.savefig('feat_gain_imp_ndcg_03.eps')


if __name__ == '__main__':
    plt.set_loglevel("info")
    # cols = ['param_bagging_freq', 'param_boosting_type', 'param_feature_fraction',
    #         'param_is_unbalance', 'param_learning_rate', 'param_max_bin',
    #         'param_metric', 'param_min_data_in_leaf', 'param_num_iterations',
    #         'param_num_leaves', 'param_objective', 'param_reg_alpha',
    #         'param_reg_lambda', 'param_subsample', 'param_subsample_for_bin',
    #         'param_verbose', 'params', 'split0_test_score', 'split1_test_score',
    #         'split2_test_score', 'split3_test_score', 'split4_test_score',
    #         'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_train_score']
    # cv_res = 'cv_results.pkl'
    # cv_res = 'full_cv_result.pkl'
    # df = pd.read_pickle(cv_res)
    # asd = 1
    z = NestedCrossVal(pd.read_pickle('data/data_df.pkl'))
    z.outer_evaluation()
    print_best_params(25)
    print_data_summary()
    plot_feature_importance()

"""
Best test AUC 0.791345
{'bagging_freq': 20, 'boosting_type': 'gbdt', 'feature_fraction': 0.25,
'is_training_metric': False, 'is_unbalance': False, 'learning_rate': 0.1, 'max_bin': 511, 'metric': 'auc',
'min_data_in_leaf': 10, 'n_estimators': 10, 'num_iterations': 50, 'num_leaves': 10, 'objective': 'binary',
'reg_alpha': 0.75, 'reg_lambda': 0.75, 'subsample': 0.8, 'subsample_for_bin': 100000, 'verbose': 1}

Best dart test AUC 0.785872
{'bagging_freq': 20, 'boosting_type': 'dart', 'feature_fraction': 0.25, 'importance_type': 'split',
'is_training_metric': False, 'is_unbalance': False, 'learning_rate': 0.1, 'max_bin': 511, 'metric': 'auc',
'min_data_in_leaf': 10, 'n_estimators': 10, 'num_iterations': 50, 'num_leaves': 31, 'objective': 'binary',
'reg_alpha': 0.75, 'reg_lambda': 0.75, 'subsample': 0.8, 'subsample_for_bin': 100000, 'verbose': 1}

"""
