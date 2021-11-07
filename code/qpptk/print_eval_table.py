from qpptk import Config
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from qpptk import add_topic_to_qdf

EASY_SYSTEM_TOPICS = {364, 420, 400}
EASY_USER_TOPICS = {364, 420, 393, 442, 385}
MEDIUM_SYSTEM_TOPICS = {385, 416}
HARD_SYSTEM_TOPICS = {393, 442, 440}
HARD_USER_TOPICS = {400, 416, 440}


def read_ap_files(raw_files, ir_metric='ap@1000'):
    results = []
    for _file in raw_files:
        collection, _ = _file.rsplit('/', 1)[1].replace('.' + ir_metric, '').split('_', 1)
        results.append(
            pd.read_table(_file, delim_whitespace=True, names=['qid', collection.split('-', 2)[2]], index_col=0))
    df = pd.concat(results, axis=1)
    df.index.name = 'qid'
    return df


def read_prediction_files(raw_files, ret_method=None):
    def concat_df(results_dic):
        results = {}
        for k, v in results_dic.items():
            _df = pd.concat(v, axis=1)
            results[k] = _df
        df = pd.concat(results)
        df.index.names = ['collection', 'topic', 'qid']
        return df

    _results = defaultdict(list)
    for _file in raw_files:
        collection, method, predictor = _file.rsplit('/', 1)[1].replace('.pre', '').split('_', 2)
        collection = collection.split('-', 2)[2]
        if method == ret_method:
            _results[collection].append(pd.read_table(_file, delim_whitespace=True, names=['topic', 'qid', predictor],
                                                      index_col=['topic', 'qid']))
    return concat_df(_results)


def plot_correlations(eval_files):
    result = []
    for ev_file in eval_files:
        result.append(pd.read_pickle(ev_file))
    df = pd.concat(result, axis=1)
    assert len(df.T) == 55
    # sort the rows by the median value of a row (predictor)
    df = df.reindex(df.median(0).sort_values().index, axis=1)
    df = df.reindex(df.median(1).sort_values().index, axis=0)
    plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
    fig = df.T.boxplot()
    # fig = sns.boxplot(x="variable", y="value", data=pd.melt(df.T), palette=sns.diverging_palette(220, 20, n=16))
    # plt.yticks(np.arange(fig.get_yticks().min(), fig.get_yticks().max(), 0.01))
    plt.ylabel(METHOD.capitalize(), fontsize=16)
    plt.xlabel('Predictor', fontsize=16)
    plt.title(f'{QUERIES.capitalize()} queries')
    plt.tight_layout()
    plt.show()


def __save_best_worst_df_by_metric(ap_files, prediction_files):
    ap_df = read_ap_files(ap_files, ir_metric=IR_METRIC)
    predictions_df = read_prediction_files(prediction_files, RETM)
    worst_collection = ap_df.mean().sort_values().head(1).index[0]
    best_collection = ap_df.mean().sort_values().tail(1).index[0]
    # print(f'Worst {worst_collection}')
    # print(f'Best {best_collection}')
    # exit()
    worst_pre_df = predictions_df.loc[worst_collection].reset_index('topic', drop=True)
    # worst_pre_df.columns = pd.MultiIndex.from_product([['Predictors'], worst_pre_df.columns])
    best_pre_df = predictions_df.loc[best_collection].reset_index('topic', drop=True)
    # best_pre_df.columns = pd.MultiIndex.from_product([['Predictors'], best_pre_df.columns])
    # worst_pre_df[('ap', worst_collection)] = ap_df[worst_collection]
    worst_pre_df[IR_METRIC] = ap_df[worst_collection]
    # best_pre_df[('ap', best_collection)] = ap_df[best_collection]
    best_pre_df[IR_METRIC] = ap_df[best_collection]
    # Dropping un-judged topic 672
    worst_pre_df = worst_pre_df.dropna()
    best_pre_df = best_pre_df.dropna()
    print(f'saving worst collection {worst_collection}')
    worst_pre_df.to_pickle(f'{RETM}_worst_collection_df.pkl')
    print(f'saving best collection {best_collection}')
    best_pre_df.to_pickle(f'{RETM}_best_collection_df.pkl')


def plot_scatters(ap_files, prediction_files):
    __save_best_worst_df_by_metric(ap_files, prediction_files)
    # worst_pre_df.merge(ap_df[worst_collection], left_on='qid', right_index=True)
    # best_pre_df = predictions_df.loc[best_collection].merge(ap_df[best_collection], left_on='qid',
    #                                                         right_index=True).reset_index('topic', drop=True)
    worst_pre_df = pd.read_pickle(f'{RETM}_worst_collection_df.pkl').rank(pct=True)
    best_pre_df = pd.read_pickle(f'{RETM}_best_collection_df.pkl').rank(pct=True)
    # sns.set(style="ticks", color_codes=True)
    sns.pairplot(worst_pre_df, kind='reg', markers='*')
    # plt.show()
    plt.savefig(f'{RETM}_worst_with_reg_{IR_METRIC}_pct.pdf', bbox_inches='tight')
    # sns.pairplot(worst_pre_df, markers='*')
    # plt.savefig(f'worst_wo_reg_{IR_METRIC}.pdf', bbox_inches='tight')
    # sns.pairplot(worst_pre_df, kind='reg', y_vars=['ap'],
    #              x_vars=worst_pre_df.columns[worst_pre_df.columns != 'ap'])
    # plt.show()
    sns.pairplot(best_pre_df, kind='reg', markers='*')
    # plt.show()
    plt.savefig(f'{RETM}_best_with_reg_{IR_METRIC}_pct.pdf', bbox_inches='tight')
    # sns.pairplot(best_pre_df, markers='*')
    # plt.savefig(f'best_wo_reg_{IR_METRIC}.pdf', bbox_inches='tight')
    # sns.pairplot(best_pre_df, kind='reg', y_vars=[best_collection],
    #              x_vars=best_pre_df.columns[best_pre_df.columns != best_collection])
    # plt.show()


def plot_raw_scores(raw_files, title_only=False):
    _df = read_ap_files(raw_files)
    if title_only:
        _df = _df.loc[_df.index.str.contains('-50-1')]
        title_queries = _df.index
        assert len(title_queries) == 249, 'wrong number of title queries in ROBUST ap file'
    assert len(_df.T) == 55
    # sort the rows by the median value of a row (predictor)
    _sr = _df.mean().sort_values()
    plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
    title = 'Title queries' if title_only else 'All queries'
    fig = sns.distplot(_sr, rug=True)
    plt.ylabel('Density', fontsize=16)
    plt.xlabel('MAP', fontsize=16)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # _sr.loc[(slice(None), 364, slice(None)), :].reset_index().rename({'level_0': 'collection'}, axis=1).drop(
    #     'topic', 1).set_index(['qid', 'collection']).corr('spearman')


def _plot_raw_scores(raw_files, topics_set, graph_title):
    def plot_boxplot(_df, _topic):
        plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        _df.boxplot()
        plt.title(f'{graph_title} #{_topic}')
        plt.tight_layout()
        plt.show()

    pre_df = read_prediction_files(raw_files, 'PRE').loc[(slice(None), topics_set, slice(None)), :]
    post_df = read_prediction_files(raw_files, 'QL').loc[(slice(None), topics_set, slice(None)), :]
    for topic, df in pre_df[['max-var', 'avg-var', 'avg-idf', 'max-idf']].groupby('topic'):
        plot_boxplot(df, topic)
    for topic, df in post_df[['smv', 'nqc', 'uef-smv', 'uef-nqc']].groupby('topic'):
        plot_boxplot(df, topic)
    for topic, df in post_df[['clarity', 'wig', 'uef-clarity', 'uef-wig']].groupby('topic'):
        plot_boxplot(df, topic)


def calc_ap_correlations_for_topic(ap_files, topic):
    _df = read_ap_files(ap_files)
    _df = add_topic_to_qdf(_df)
    ap_df = _df.loc[_df['topic'].astype(int) == topic].set_index(['topic', 'qid'])
    corr = ap_df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, n=10, center="light", as_cmap=True)
    max = corr.max().max()
    min = corr.min().min()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=min, vmax=max, center=np.mean([max, min]), square=True,
                linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(f'Topic #{topic} AP correlations')
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    plt.show()


if __name__ == '__main__':
    plt.set_loglevel("info")
    # METHOD = 'spearman'
    METHOD = 'pearson'
    # METHOD = 'kendall'
    # QUERIES = 'title'
    RETM = 'PRE'
    QUERIES = 'all'
    IR_METRIC = 'R-precision'
    results_dir = Config.RESULTS_DIR
    eval_files = glob(f'{results_dir}/*eval_{METHOD}_{QUERIES}_queries.pkl')
    ap_files = glob(f'{results_dir}/*_QL.{IR_METRIC}')
    prediction_files = glob(f'{results_dir}/*.pre')
    # plot_correlations(eval_files)
    # plot_raw_scores(ap_files, title_only=True)
    # plot_raw_scores(ap_files, title_only=False)
    # _plot_raw_scores(prediction_files, EASY_SYSTEM_TOPICS.intersection(EASY_USER_TOPICS),
    #                  graph_title='SE & UE - Easy Topic')
    # _plot_raw_scores(prediction_files, HARD_SYSTEM_TOPICS.intersection(HARD_USER_TOPICS),
    #                  graph_title='SH & UH - Hard Topic')
    # calc_ap_correlations_for_topic(ap_files, EASY_SYSTEM_TOPICS.intersection(EASY_USER_TOPICS).pop())
    # calc_ap_correlations_for_topic(ap_files, HARD_SYSTEM_TOPICS.intersection(HARD_USER_TOPICS).pop())
    # __fun(ap_files, prediction_files)
    plot_scatters(ap_files, prediction_files)
