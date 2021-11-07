import os
import sys
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score
from itertools import combinations, permutations

from qpptk import Config, ensure_file, add_topic_to_qdf, ensure_dir, overlap_coefficient
from qpptk.global_manager import initialize_text_queries, initialize_ciff_queries


# plt.switch_backend('Qt5Agg')

def filter_robust_title_queries(_df):
    _df = _df.loc[_df.index.str.contains('-50-1')]
    # assert len(_df) == 249, 'wrong numbers of title queries in ROBUST eval file'
    return _df


def get_queries_object():
    queries_path, queries_type = (Config.TEXT_QUERIES, 'text') if Config.TEXT_QUERIES else \
        (Config.CIFF_QUERIES, 'ciff')
    if queries_path is None:
        raise AssertionError('No queries file was specified')
    return initialize_text_queries(queries_path) if queries_type == 'text' else initialize_ciff_queries(queries_path)


def set_index():
    index_path, index_type = (Config.INDEX_DIR, 'text') if Config.INDEX_DIR else (Config.CIFF_INDEX, 'ciff')
    if index_path is None:
        raise AssertionError('No index was specified')
    return index_path, index_type


def print_samples_ratios(data_df):
    logger.info(f'{data_df["label"].sum()} samples out of {len(data_df)} were marked positive')
    logger.info(f'Ratio of positive samples in the data: {data_df["label"].sum() / len(data_df) :.2f}')
    logger.info(f"Ratio of topics with positive samples: "
                f"{data_df[['topic', 'label']].groupby('topic').any().sum()[0] / len(data_df.groupby('topic')) :.2f}")


def read_eval_df(prefix_path, ir_metric):
    eval_file = ensure_file(f"{prefix_path}_QL.{ir_metric}")
    eval_df = pd.read_csv(eval_file, delim_whitespace=True, names=['qid', ir_metric], index_col=0)
    return eval_df.drop(DUPLICATED_QIDS, errors='ignore')
    # return eval_df


def generate_labeled_df(prefix_path, ir_metric, threshold):
    eval_df = read_eval_df(prefix_path, ir_metric)
    # return eval_df
    _threshold = eval_df.quantile(threshold)[0]
    logger.info(f"Threshold for bad queries was set to {_threshold:.3f}")
    # This will label all variants above the threshold as 0, and less eq than as 1
    labeled_df = eval_df.where(eval_df.values <= _threshold, 0).mask(eval_df.values <= _threshold, 1).rename(
        columns={ir_metric: 'label'})
    labeled_df = add_topic_to_qdf(labeled_df).set_index('qid')
    assert labeled_df['label'].sum() > 0, 'No samples were labeled as positive'
    print('\nSamples ratios in the data')
    print_samples_ratios(labeled_df)
    return labeled_df, _threshold


def read_lgbm_prediction_files(ir_metric, threshold=None):
    if threshold:
        try:
            _file = ensure_file(f'{ir_metric}_q-{threshold:.1f}_LGBM.pre')
        except FileNotFoundError as er:
            sys.exit(er)
        return pd.read_csv(_file, delim_whitespace=True, names=['qid', 'LGBM'], index_col=['qid'])
    else:
        predictors = glob(f'*{ir_metric}*' + '.pre')
    _results = []
    for _file in predictors:
        _ir_metric, threshold, predictor = _file.replace('.pre', '').split('_', 2)
        _results.append(
            pd.read_csv(_file, delim_whitespace=True, names=['qid', f'Q-{threshold}'], index_col=['qid']))
    return pd.concat(_results, axis=1)


def read_prediction_files(prefix_path, r_type='all'):
    if r_type == 'all':
        post_ret_predictors = glob(prefix_path + '_QL*.pre')
        pre_ret_predictors = glob(prefix_path + '*PRE*')
        predictors = pre_ret_predictors + post_ret_predictors
    elif r_type.lower() == 'pre':
        predictors = glob(prefix_path + '*PRE*')
    else:
        predictors = glob(prefix_path + '_QL*.pre')
    _results = []
    for _file in predictors:
        collection, method, predictor = _file.rsplit('/', 1)[1].replace('.pre', '').rsplit('_', 2)
        _results.append(
            pd.read_csv(_file, delim_whitespace=True, names=['topic', 'qid', predictor], index_col=['topic', 'qid']))
    return pd.concat(_results, axis=1).drop(672, errors='ignore').drop(DUPLICATED_QIDS, level=1, errors='ignore')


def df_to_libsvm(df: pd.DataFrame, set_name):
    data_dir = ensure_dir('data')
    x = df.drop(['qid', 'topic', 'label'], axis=1)
    y = df['label']
    dump_svmlight_file(X=x, y=y, f=f'{data_dir}/{set_name}.txt', zero_based=True)


def generate_random_col(data_df):
    x = np.arange(len(data_df))
    np.random.shuffle(x)
    return x


def plot_hard_queries(prefix_path, ir_metric):
    predictors_group = 'post'
    eval_df = read_eval_df(prefix_path, ir_metric)
    predictions_df = read_prediction_files(prefix_path, predictors_group)
    data_df = predictions_df.merge(eval_df, left_on='qid', right_on='qid')
    data_df['random'] = generate_random_col(data_df)
    diff_df = data_df.drop(ir_metric, 1).rank(pct=True, ascending=False).subtract(
        data_df[ir_metric].rank(pct=True, ascending=False), axis=0)
    print(f'Number of evaluated queries: {len(diff_df)}')
    # for predictor in diff_df.columns:
    # print(f'{predictor}:')
    # print(f'Largest differences (ranked higher by predictor): {diff_df[predictor].nlargest(10)}')
    # print(f'Smallest differences (ranked lower by predictor): {diff_df[predictor].nsmallest(10)}')
    print(f'Poor queries that were predicted as good:')
    poorest = diff_df.min(1).nlargest(330).index
    print(f'poorest topics:')

    bad_topics = add_topic_to_qdf(eval_df).nlargest(330, ir_metric).topic
    for pred in diff_df.columns:
        poor_topics = add_topic_to_qdf(diff_df)[['topic', 'qid', pred]].nsmallest(330, pred).topic
        print(pred, overlap_coefficient(set(bad_topics), set(poor_topics)))
    bad_topics = add_topic_to_qdf(eval_df).nsmallest(330, ir_metric).topic
    for pred in diff_df.columns:
        poor_topics = add_topic_to_qdf(diff_df)[['topic', 'qid', pred]].nsmallest(330, pred).topic
        print(pred, overlap_coefficient(set(bad_topics), set(poor_topics)))
    bad_queries = eval_df.nsmallest(330, ir_metric).index
    # print(overlap_coefficient(set(bad_topics), set(add_topic_to_qdf(diff_df.min(1).nlargest(15)).topic)))
    print('poorest queries:')
    print(overlap_coefficient(set(bad_queries), set(diff_df.min(1).nlargest(330).index)))
    # plot_qq_plot(diff_df, False)
    plot_predictors_errors(diff_df.pow(2), title=f'{predictors_group.capitalize()}-Retrieval squared difference',
                           save_figure=False)
    # plot_hard_topics(diff_df.loc[diff_df.loc[poorest].T.median().sort_values().index].T, ir_metric)
    # diff_df = add_topic_to_qdf(diff_df)
    # sns.pairplot(diff_df.set_index('qid').groupby('topic').median())
    # sns.pairplot(diff_df, markers='*')
    # plt.show()
    # diff_df.to_pickle(f'diff_df_{ir_metric}.pkl')

    # exit()


def plot_hard_topics(topics_df, ir_metric):
    plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig, ax = plt.subplots()
    # sns.set(font='serif', font_scale=2)
    fig.suptitle(f'Top Ranking Errors by Query {ir_metric}')
    # Create the boxplot and store the resulting python dictionary
    boxes = ax.boxplot(topics_df.values)
    ax.set_xticklabels(topics_df.columns, rotation=45)
    make_flier_labels(ax, boxes, topics_df)
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'top_15_ranking_errors_{ir_metric}.eps')


def make_flier_labels(ax, boxplots, df):
    for i, box in enumerate(boxplots['fliers']):
        fly = box
        # The x position of the flyers
        x_pos = boxplots['medians'][i].get_xdata()

        # Add text a horizontal offset as a fraction of the width of the box
        x_off = 0.10 * (x_pos[1] - x_pos[0])
        sr = df.iloc[:, i]
        eps = 0.0001
        for flier in fly.get_ydata():
            predictor = sr.loc[(flier - eps <= sr) & (sr <= flier + eps)].index[0]
            ax.annotate(f' {predictor}', (1 + i + x_off, flier), xytext=(0, -3), textcoords="offset points",
                        ha='center',
                        va='top')


def generate_baselines_results(data_df):
    auc_res = []
    for predictor in data_df.drop(['topic', 'label'], 1).columns:
        auc_res.append((predictor, roc_auc_score(data_df['label'], -data_df[predictor])))
        # plot_roc(data_df['label'], -data_df[predictor], predictor)
    return pd.DataFrame(auc_res, columns=['Predictor', 'AUC-ROC']).set_index('Predictor')


def print_baselines_table_by_threshold(prefix_path, ir_metric):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = []
    for t in thresholds:
        data_df, threshold = generate_data_df(prefix_path, ir_metric, t)
        # nc = NestedCrossVal(data_df=data_df)
        # nc.outer_evaluation(f'{ir_metric}_q-{t:.1f}')
        lgbm_score = roc_auc_score(data_df['label'], read_lgbm_prediction_files(ir_metric, t))
        _df = generate_baselines_results(data_df).rename({'AUC-ROC': f'Q={t:.1f} T={threshold:.3f}'}, axis=1)
        _df.loc['LightGBM'] = lgbm_score
        res.append(_df)
    df = pd.concat(res, 1)
    df = df.reindex(df.mean(1).sort_values().index)
    df.columns = pd.MultiIndex.from_product([[ir_metric], df.columns])
    print(df.to_latex(multicolumn=True, float_format='%.3f', column_format='lccccc'))
    return df


def calc_correlations_df(predictions_df: pd.DataFrame, eval_df, title_only=False):
    results_pearson = []
    results_kendall = []
    if title_only:
        eval_df = filter_robust_title_queries(eval_df)
    for pred in predictions_df.columns:
        results_pearson.append(
            (pred, eval_df.merge(predictions_df[pred], left_index=True, right_on='qid').corr('pearson').values[0][1]))
        results_kendall.append(
            (pred, eval_df.merge(predictions_df[pred], left_index=True, right_on='qid').corr('kendall').values[0][1]))

    p_corrs_df = pd.DataFrame(results_pearson, columns=['predictor', 'pearson correlation']).set_index('predictor')
    k_corrs_df = pd.DataFrame(results_kendall, columns=['predictor', 'kendall correlation']).set_index('predictor')
    corrs_df = p_corrs_df.merge(k_corrs_df, left_index=True, right_index=True).sort_values('pearson correlation')
    corrs_df = corrs_df.assign(
        rank_diff=corrs_df['pearson correlation'].rank() - corrs_df['kendall correlation'].rank())
    print(corrs_df.to_latex(float_format='%.3f', column_format='lccc'))
    return corrs_df


def generate_data_df(prefix_path, ir_metric, quantile_threshold):
    labeled_df, threshold = generate_labeled_df(prefix_path, ir_metric, quantile_threshold)
    predictions_df = read_prediction_files(prefix_path)
    return predictions_df.merge(labeled_df, left_on='qid', right_on='qid'), threshold


def plot_dist(data_sr, title='', xlabel=None, ylabel='Density', save_figure=False):
    """
    KDE = kernel density estimation (KDE) is a non-parametric way to estimate the
    probability density function of a random variable.
    """
    titles = {'ndcg@10': 'nDCG@10', 'ndcg@100': 'nDCG@100', 'ap@1000': 'AP@1000'}
    # sns.set_color_codes("muted")  # {deep, muted, pastel, dark, bright, colorblind}
    # sns.set(font='serif', font_scale=2)
    # sns.set_style("white")
    # sns.set_palette("PuBuGn_d")
    # plt.figure(figsize=(7, 4), dpi=100)
    sns.distplot(data_sr, hist=False, rug=True, kde=True,
                 # fit_kws={"lw": 2, "label": "FIT"},
                 rug_kws={'color': '#2F4C56', "alpha": 0.2},
                 kde_kws={"lw": 2, "label": "KDE", "bw": "silverman"},
                 hist_kws={"histtype": "bar", "lw": 1, "alpha": 0.5})  # fit=stats.gamma to fit gamma function
    # plt.ylabel(ylabel)
    # plt.xlabel(xlabel)
    # plt.title(titles.get(title, title))
    # plt.tight_layout()
    sns.despine(left=True)
    if save_figure:
        plt.savefig(f'{title}_dist.pdf')
    else:
        return
        # plt.show()


def plot_eval_metrics_dist(prefix_path):
    def _poor_topics(_eval_df, _ir_metric, _n):
        # return add_topic_to_qdf(_eval_df.nsmallest(_n, _ir_metric)).topic
        return add_topic_to_qdf(_eval_df.nsmallest(_n, _ir_metric)).qid

    # save_figures = True
    save_figures = False
    ir_metric = 'ap@1000'
    eval_df = read_eval_df(prefix_path, ir_metric)
    n = len(eval_df) // 10
    plot_dist(eval_df, title=ir_metric, save_figure=save_figures)
    poor_ap_topics = set(_poor_topics(eval_df, ir_metric, n))
    ir_metric = 'ndcg@100'
    eval_df.corrwith(read_eval_df(prefix_path, ir_metric))
    eval_df = read_eval_df(prefix_path, ir_metric)
    plot_dist(eval_df, title=ir_metric, save_figure=save_figures)
    poor_ndcg_100_topics = set(_poor_topics(eval_df, ir_metric, n))
    ir_metric = 'ndcg@10'
    eval_df = read_eval_df(prefix_path, ir_metric)
    plot_dist(eval_df, title=ir_metric, save_figure=save_figures)
    poor_ndcg_10_topics = set(_poor_topics(eval_df, ir_metric, n))
    print(f'{n} is ~10% of the data, printing coefficients of the {n} worst topics:')
    print('Overlap coefficient of AP-nDCG@100:')
    print(overlap_coefficient(poor_ap_topics, poor_ndcg_100_topics))
    print('Overlap coefficient of AP-nDCG@10:')
    print(overlap_coefficient(poor_ap_topics, poor_ndcg_10_topics))
    print('Overlap coefficient of nDCG@100-nDCG@10:')
    print(overlap_coefficient(poor_ndcg_100_topics, poor_ndcg_10_topics))


def plot_predictors_errors(diff_df: pd.DataFrame, title='', subplots=True, save_figure=False):
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "sans-serif",
    #     "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.figure(figsize=(16, 9), dpi=100)
    ax = diff_df[diff_df.median().sort_values().index].boxplot()
    plt.xticks(rotation=45)
    if save_figure:
        plt.savefig(f'prediction_errors_box.pdf')
    else:
        plt.show()
    if subplots:
        plt.figure(figsize=(16, 9), dpi=100)
        for pred in diff_df.median().sort_values().index:
            sns.distplot(diff_df[pred], hist=False, rug=True, kde=True, label=pred,
                         # fit_kws={"lw": 2, "label": "FIT"},
                         # rug_kws={'color': '#2F4C56', "alpha": 0.2},
                         kde_kws={"lw": 2, "label": "KDE", "clip": (0, 1), "bw": "silverman"})
        plt.title(title)
        plt.xlabel('Ranks Difference')
        plt.ylabel('Density')
        plt.grid()
        if save_figure:
            plt.savefig(f'prediction_errors_dist_{title.replace(" ", "_")}.pdf')
        else:
            plt.show()
    else:
        # _df = diff_df.melt()
        # g = sns.FacetGrid(_df, col="variable", col_wrap=5, height=1.5)
        # g = g.map(sns.distplot, "value")
        # plt.show()
        # exit()
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "sans-serif",
        #     "font.sans-serif": ["Helvetica"]})
        for pred in diff_df.columns:
            sns.distplot(diff_df[pred], hist=False, rug=True, kde=True, label=pred,
                         fit_kws={"lw": 2, "label": "FIT"},
                         rug_kws={'color': '#2F4C56', "alpha": 0.2},
                         kde_kws={"lw": 2, "label": "KDE", "clip": (0, 1), "bw": "silverman"},
                         hist_kws={"histtype": "bar", "lw": 1, "alpha": 0.5})
            plot_dist(diff_df[pred], title=pred + ' Error', save_figure=save_figure)


def plot_qq_plot(diff_df, save_figure=False):
    for pred in diff_df.columns:
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        # sns.set(font='serif')
        # sns.set_style("white")
        # sns.set_palette("PuBuGn_d")
        plt.figure(figsize=(7, 4), dpi=100)
        fig = sm.qqplot(diff_df[pred], line='s', color='#2F4C56')
        fig.set_size_inches((7, 4))
        fig.set_dpi(100)
        fig.tight_layout()
        if save_figure:
            fig.savefig(f'prediction_errors_box.pdf')
        else:
            fig.show()


def plot_predictors_dist(prefix_path):
    predictions_df = read_prediction_files(prefix_path, 'pre').reset_index('topic', drop=True)
    for pred in predictions_df.columns:
        plot_dist(predictions_df[pred], title=pred, save_figure=False)


def eval_to_rank_map(eval_df, predictions_df, ranks_method):
    ir_metric = eval_df.columns[0]
    rank_ap_dict = eval_df.assign(rank=eval_df.rank(ascending=False, method=ranks_method)).set_index('rank')[
        ir_metric].to_dict()
    df = predictions_df.rank(ascending=False, method=ranks_method).applymap(
        lambda x: rank_ap_dict.get(x, find_closest_key(x, rank_ap_dict)))
    diff_df = df.subtract(eval_df[ir_metric], axis=0)
    return diff_df


def find_closest_key(val, dict_val):
    return dict_val.get(min(dict_val.keys(), key=lambda x: abs(x - val)))


def test_for_normality(data_sr, alpha=5e-2, title=None):
    """
    Noramlity test that based on D'Agostino and
    Pearson's test that combines skew and kurtosis to
    produce an omnibus test of normality
    :param data_sr:
    :rtype: bool
    :return True if data is normally distributed False otherwise
    """
    if len(data_sr) >= 20:
        stat, p = stats.normaltest(data_sr)
    else:
        logger.info('The sample size is smaller than 20, using Shapiro test')
        stat, p = stats.shapiro(data_sr)
    logger.debug(f'p value = {p}')
    if p < alpha:  # null hypothesis: data_sr comes from a normal distribution
        logger.info(
            f'{title.capitalize()} is not normally distributed with sig level {alpha}') if title else logger.info(
            f'The result is not normally distributed with sig level {alpha}')
        return False
    else:
        logger.info(f'{title.capitalize()} is normally distributed with sig level {alpha}') if title else logger.info(
            f'The result is normally distributed with sig level {alpha}')
        return True


def test_variance_equality(*args, alpha=5e-2):
    """
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
    Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances.
    :param args:
    :param alpha:
    :return:
    """
    stat, p = stats.levene(args)
    logger.debug(f'p value = {p}')
    if p < alpha:  # null hypothesis: data_sr comes from a normal distribution
        logger.info(f'The results does not have equal variances with sig level {alpha}')
        return False
    else:
        logger.info(f'The results do have equal variances with sig level {alpha}')
        return True


def test_eval_measures(ir_metric, title_only=False, pct=True):
    predictors_type = 'all'
    correlation_method = 'kendall'
    # for corpus in {'robust04', 'core18', 'core17'}:
    for corpus in {'robust04'}:
        all_eval_files = glob(f'{results_dir}/{corpus}*{ir_metric}')
        correlations = []
        agreement_rate = []
        avg_corr = []
        avg_abs_mean = []
        for eval_file in all_eval_files:
            pipe, _metric = eval_file.split('/')[-1].split('.')
            prefix = pipe.rsplit('_', 1)[0]
            prefix_path = os.path.join(results_dir, prefix)
            predictions_df = read_prediction_files(prefix_path, predictors_type).reset_index('topic', drop=True)
            eval_df = read_eval_df(prefix_path, ir_metric)
            if title_only:
                predictions_df = filter_robust_title_queries(predictions_df)
                eval_df = filter_robust_title_queries(eval_df)
            corr_res = predictions_df.corrwith(eval_df[ir_metric], 0, method=correlation_method)
            mean_diff_df = predictions_df.rank(pct=pct, ascending=False, method='average').subtract(
                eval_df[ir_metric].rank(pct=pct, ascending=False, method='average'), axis=0)
            for topic, _df in add_topic_to_qdf(mean_diff_df.abs()).set_index('qid').groupby('topic'):
                _df.drop('topic',axis=1).boxplot()
            dense_diff_df = predictions_df.rank(pct=pct, ascending=False, method='dense').subtract(
                eval_df[ir_metric].rank(pct=pct, ascending=False, method='dense'), axis=0)
            first_diff_df = predictions_df.rank(pct=pct, ascending=False, method='first').subtract(
                eval_df[ir_metric].rank(pct=pct, ascending=False, method='first'), axis=0)
            diff_dense = dense_diff_df.mean().corr(corr_res, method=correlation_method)
            sqr_mean = mean_diff_df.pow(2).mean().corr(corr_res, method=correlation_method)
            sqr_dense = dense_diff_df.pow(2).mean().corr(corr_res, method=correlation_method)
            abs_mean = mean_diff_df.abs().mean().corr(corr_res, method=correlation_method)
            abs_dense = mean_diff_df.abs().mean().corr(corr_res, method=correlation_method)
            correlations.append(
                {'dense-raw': diff_dense, 'mean-sqr': sqr_mean, 'dense-sqr': sqr_dense, 'mean-abs': abs_mean,
                 'dense-abs': abs_dense})
            agreement_rate.append({'mean-abs': pairwise_agreement(corr_res, mean_diff_df.abs()),
                                   'mean-sqr': pairwise_agreement(corr_res, mean_diff_df.pow(2)),
                                   'dense-raw': pairwise_agreement(corr_res, dense_diff_df),
                                   'dense-abs': pairwise_agreement(corr_res, dense_diff_df.abs()),
                                   'dense-sqr': pairwise_agreement(corr_res, dense_diff_df.pow(2)),
                                   'first-sqr': pairwise_agreement(corr_res, first_diff_df.pow(2)),
                                   'first-abs': pairwise_agreement(corr_res, first_diff_df.abs())})
            avg_corr.append(corr_res)
            avg_abs_mean.append(mean_diff_df.abs().mean())
        avg_corr_df = pd.concat(avg_corr, axis=1)
        title = 'title' if title_only else 'all'
        # avg_corr_df.to_pickle(f'{results_dir}/{corpus}_{predictors_type}_{title}_kendall_scores.pkl')
        avg_abs_mean_df = pd.concat(avg_abs_mean, axis=1)
        avg_abs_mean_df.to_pickle(f'{results_dir}/{corpus}_{predictors_type}_{title}_mrd_scores.pkl')
        print(corpus)
        comp_df = pd.DataFrame(
            {'AbsRankDiff': avg_abs_mean_df.mean(1), 'Correlation': avg_corr_df.mean(1)}).sort_values('Correlation')
        print(comp_df.rename(LatexMacros).to_latex(float_format='%.3f', caption=f'{corpus} queries',
                                                   escape=False))
        corrs_df = pd.DataFrame(correlations)
        agreement_rate_df = pd.DataFrame(agreement_rate)
        df = pd.DataFrame({'Mean': corrs_df.mean(), 'STD': corrs_df.std()})
        _df = pd.DataFrame({'Mean': agreement_rate_df.mean(), 'STD': agreement_rate_df.std()})
        comb_df = pd.concat([_df, df], axis=1)
        print(comb_df.to_latex(caption=f'{corpus}', float_format='%.3f'))


def pairwise_agreement(correlations_sr, diff_df):
    result = []
    for p1, p2 in combinations(diff_df.columns, 2):
        p1_corr = correlations_sr.loc[p1]
        p2_corr = correlations_sr.loc[p2]
        if p1_corr > p2_corr:
            corr_pick = p1
        elif p1_corr < p2_corr:
            corr_pick = p2
        else:
            corr_pick = 'tie'
        if diff_df[p1].mean() < diff_df[p2].mean():
            diff_pick = p1
        elif diff_df[p1].mean() > diff_df[p2].mean():
            diff_pick = p2
        else:
            diff_pick = 'tie'
        result.append(corr_pick == diff_pick)
    return np.mean(result)


def one_way_anova(corpus='robust04', predictors_type='all', queries_group='title', save_figure=True):
    meanprops = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue"}
    sns.set(font='serif', context='poster', style="whitegrid")
    corr_file = f'{results_dir}/{corpus}_{predictors_type}_{queries_group}_kendall_scores.pkl'
    msqr_file = f'{results_dir}/{corpus}_{predictors_type}_{queries_group}_mrd_scores.pkl'
    corr_df = pd.read_pickle(corr_file).rename(PlotNames)
    msqr_df = pd.read_pickle(msqr_file).rename(PlotNames)
    _cdf = corr_df.T.melt(var_name='predictor', value_name='Correlation')
    _rdf = msqr_df.T.melt(var_name='predictor', value_name='sMARE')
    fig, ax = plt.subplots(figsize=(12, 9))
    if save_figure:
        title = None
        file_name = f'{corpus}_{predictors_type}_predictors_{queries_group}_kendall_scores.pdf'
    else:
        title = f'{corpus}_{predictors_type}_predictors_{queries_group}_kendall_scores'.replace('_', ' ').capitalize()
    sns.boxplot('Correlation', 'predictor', data=_cdf, order=corr_df.median(1).sort_values(ascending=False).index,
                ax=ax, showmeans=True, meanprops=meanprops)
    plt.title(title)
    plt.ylabel(None)
    # plt.xlabel('Correlation')
    plt.tight_layout()
    if save_figure:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
    if save_figure:
        title = None
        file_name = f'{corpus}_{predictors_type}_predictors_{queries_group}_mrd_scores.pdf'
    else:
        title = f'{corpus}_{predictors_type}_predictors_{queries_group}_mrd_scores'.replace('_', ' ').capitalize()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.boxplot('sMARE', 'predictor', data=_rdf, order=msqr_df.median(1).sort_values(ascending=True).index, ax=ax,
                showmeans=True, meanprops=meanprops)
    plt.title(title)
    plt.xlabel('sMARE$_{AP}$')
    plt.ylabel(None)
    plt.tight_layout()
    if save_figure:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
    # print(stats.f_oneway(*corr_df.values))
    # _df = pd.concat([_cdf.assign(type='sMARE'), _rdf.assign(type='Correlation')])
    # sns.catplot('value', 'predictor', col='type', kind='box', data=_df, order=corr_df.median(1).sort_values().index)
    print(stats.f_oneway(*msqr_df.values))


def plot_rank_diff_prob(n):
    _df = pd.DataFrame(permutations(range(1, n + 1), n)).T
    print('Created df')
    tdf = pd.Series(range(1, n + 1))
    print('Created tdf')
    diff_df = _df.subtract(tdf, axis=0)
    print('Created diff_df')
    s = len(diff_df.T)
    print(diff_df.abs().mean().value_counts() / s)
    sns.distplot(diff_df.abs().mean(), rug=True, kde=False)
    plt.show()


def main():
    plt.set_loglevel("info")
    index_path, index_type = set_index()
    prefix = '_'.join(index_path.split('/')[-2:]) if index_type == 'text' else \
        index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    prefix_path = os.path.join(results_dir, prefix)
    # plot_eval_metrics_dist(prefix_path)
    # plot_predictors_dist(prefix_path)
    # exit()
    ir_metric = 'ap@1000'
    # ir_metric = 'ndcg@10'
    # eval_df = read_eval_df(prefix_path, ir_metric)
    # lgbm_df = read_lgbm_prediction_files(ir_metric, 0.1)
    # plot_dist(lgbm_df)
    # exit()
    # corr_df = calc_correlations_df(read_prediction_files(prefix_path, 'all'), eval_df)
    # auc_df = print_baselines_table_by_threshold(prefix_path, ir_metric)
    # print(corr_df)
    # exit()
    # plot_rank_diff_prob(10)
    test_eval_measures(ir_metric, title_only=False, pct=True)
    # one_way_anova()
    # plot_hard_queries(prefix_path, ir_metric)
    exit()
    data_df, threshold = generate_data_df(prefix_path, ir_metric, quantile_threshold=0.3)
    logger.info(f'The labeling threshold was set to {threshold} {ir_metric}')
    data_df.to_pickle('data/data_df.pkl')
    topics = data_df['topic'].unique()
    np.random.seed(321)
    msk = np.random.rand(len(topics)) < 0.7
    train = topics[msk]
    test = topics[~msk]
    train_df = data_df.loc[data_df['topic'].isin(train)]
    test_df = data_df.loc[data_df['topic'].isin(test)]
    print('\nTrain set ratios')
    print_samples_ratios(train_df)
    print('\nTest set ratios')
    print_samples_ratios(test_df)

    train_df.to_pickle('data/train_df.pkl')
    test_df.to_pickle('data/test_df.pkl')
    df_to_libsvm(train_df.reset_index(), 'train')
    df_to_libsvm(train_df.reset_index(), 'test')


if __name__ == '__main__':
    PreRetPredictors = {'scq', 'avg-scq', 'max-scq', 'var', 'avg-var', 'max-var', 'max-idf', 'avg-idf'}
    PostRetPredictors = {'clarity', 'smv', 'nqc', 'wig', 'uef-clarity', 'uef-smv', 'uef-nqc', 'uef-wig'}
    LatexMacros = {'scq': '\\Scq', 'avg-scq': '\\avgScq', 'max-scq': '\\maxScq', 'var': '\\Var', 'avg-var': '\\avgVar',
                   'max-var': '\\maxVar', 'max-idf': '\\maxIDF', 'avg-idf': '\\avgIDF',
                   'clarity': '\\clarity', 'smv': '\\smv', 'nqc': '\\nqc', 'wig': '\\wig',
                   'uef-clarity': '\\uef{\\clarity}', 'uef-smv': '\\uef{\\smv}', 'uef-nqc': '\\uef{\\nqc}',
                   'uef-wig': '\\uef{\\wig}'}
    PlotNames = {'scq': 'SCQ', 'avg-scq': 'AvgSCQ', 'max-scq': 'MaxSCQ', 'var': 'SumVAR', 'avg-var': 'AvgVAR',
                 'max-var': 'MaxVAR', 'max-idf': 'MaxIDF', 'avg-idf': 'AvgIDF', 'clarity': 'Clarity', 'smv': 'SMV',
                 'nqc': 'NQC', 'wig': 'WIG', 'uef-clarity': 'UEF(Clarity)', 'uef-smv': 'UEF(SMV)',
                 'uef-nqc': 'UEF(NQC)', 'uef-wig': 'UEF(WIG)'}
    with open('duplicated_qids.txt') as f:
        DUPLICATED_QIDS = {line.rstrip('\n') for line in f}
    logger = Config.get_logger()
    results_dir = Config.RESULTS_DIR
    main()
