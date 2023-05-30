import os
import pprint
from os.path import join as pjoin
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.metrics import roc_curve, balanced_accuracy_score
from sklearn.metrics import auc, confusion_matrix, recall_score, precision_score
from matplotlib.pylab import rcParams

import utils


def train_val_test_split(X, X_clean_test, config):
    # Verify it's empty intersection     #TODO: add as assertion
    # X_test,y_test = X[X[config.bets_player_id_column].isin(test_id.values)] , X[X[config.bets_player_id_column].isin(test_id.values)][config.label_column_name].values
    # X_train = X[~X[config.bets_player_id_column].isin(test_id.values)]
    # X_val,y_val = X_train[X_train[config.bets_player_id_column].isin(val_id.values)] , X_train[X_train[config.bets_player_id_column].isin(val_id.values)][config.label_column_name].values
    # X_train,y_train = X_train[~X_train[config.bets_config.bets_player_id_column].isin(val_id.values)] , X_train[~X_train[config.bets_player_id_column].isin(val_id.values)][config.label_column_name].values
    # print(set(X_train.player_id) & set(X_test.player_id) |set(X_train.player_id) & set(X_val.player_id) |set(X_val.player_id) & set(X_test.player_id)  )

    val_id = X[config.bets_player_id_column].sample(
        int(np.round(X[config.bets_player_id_column].nunique() * config.user_ratio_in_val)),
        random_state=1)  # POC with random_state=1

    X_val, y_val = X[X[config.bets_player_id_column].isin(val_id.values)], \
        X[X[config.bets_player_id_column].isin(val_id.values)][config.label_column_name].values
    X_train, y_train = X[~X[config.bets_player_id_column].isin(val_id.values)], \
        X[~X[config.bets_player_id_column].isin(val_id.values)][config.label_column_name].values

    for df, split in zip([X_train, X_val, X_clean_test], ['X_train', 'X_val', 'X_test']):
        print(split)
        print('0:', df[df[config.label_column_name] == False].player_id.nunique(), '; 1:',
              df[df[config.label_column_name] == True].player_id.nunique())
        print(f"counts [0 1]: {np.unique(df[config.label_column_name], return_counts=True)[1]}")

    # Test
    X_clean_test, y_clean_test = X_clean_test.drop(
        columns=[config.label_column_name, config.bets_player_id_column]).values, X_clean_test[
        config.label_column_name].values

    # Val and Train

    y_val, y_train = X_val[config.label_column_name].values, X_train[config.label_column_name].values
    X_val, X_train = X_val.drop(columns=[config.label_column_name, config.bets_player_id_column]).values, X_train.drop(
        columns=[config.label_column_name, config.bets_player_id_column]).values

    # Train val test split - regular data - users are mixed
    # X_train, X_clean_test, y_train, y_test = train_test_split(X.values, y.values, test_size=.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, random_state=42)

    unique, counts = np.unique(y_val, return_counts=True)
    print([f"{x} : counts [0 1]: {np.unique(y, return_counts=True)[1]}" for x, y in
           zip(["Train", "Val", "Clean Test"], [y_train, y_val, y_clean_test])])

    splits = (
        X_train.astype(float), y_train.astype(float), X_val.astype(float), y_val.astype(float),
        X_clean_test.astype(float),
        y_clean_test.astype(float))

    return splits


def train_model(splits, config, test=False, save_model=True, model_filename='xgb_model', manual_model_params=False):
    """### Fit"""
    path_to_model = None
    X_train, y_train, X_val, y_val, X_clean_test, y_clean_test = splits
    positive_ratio = np.unique(y_train, return_counts=True)[1][0] / np.unique(y_train, return_counts=True)[1][
        1]  # get positive scale_ratio

    xgb_params = config.xgb_params if not manual_model_params else manual_model_params
    xgb_params['scale_pos_weight'] = positive_ratio  # add *train* calculated scale_pos_weight

    xgb_model = xgb.XGBClassifier(**xgb_params)  # if not RegTask else xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

    # make predictions
    pprint.pprint(xgb_params)  # pretty print
    print("Task is VIP") if config.learn_task_vip else print("Task is STD")
    print(f"Active Users are Labeled as '{int(config.active_user_label_1)}'")
    print("Results on Clean Test Data:")
    preds = xgb_model.predict(X_clean_test)
    tn, fp, fn, tp = confusion_matrix(y_clean_test, preds).ravel()
    print("TN", tn, "FP", fp, "FN", fn, "TP", tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f'Precision:{prec:.2f}', f'Recall: {rec:.2f}', f'F1: {2 * prec * rec / (prec + rec)}',
          f'Neg. Precision: {1 - (fn / (tn + fn))}')
    print("")
    print("Results on Train:")
    preds = xgb_model.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
    print("TN", tn, "FP", fp, "FN", fn, "TP", tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f'Precision:{prec:.2f}', f'Recall: {rec:.2f}', f'F1: {2 * prec * rec / (prec + rec)}',
          f'Neg. Precision: {1 - (fn / (tn + fn))}')

    print("Results on Validation:")
    preds = xgb_model.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    print("TN", tn, "FP", fp, "FN", fn, "TP", tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f'Precision:{prec:.2f}', f'Recall: {rec:.2f}', f'F1: {2 * prec * rec / (prec + rec)}',
          f'Neg. Precision: {1 - (fn / (tn + fn))}')

    """### Save Model"""
    if save_model:
        path = pjoin(config.data_folder_path, config.data_dir, 'models') if not manual_model_params else \
            pjoin(config.data_folder_path, config.data_dir, 'models', 'parameter_search')
        if not os.path.exists(path):
            os.makedirs(path)
        filename = model_filename + f'{config.sport_or_casino_task}'  # Casino or Sports
        filename = filename + f'_VIP_{config.active_users_sum_deposits_vip}' if config.learn_task_vip else model_filename + '_STD'
        filename = filename + 'TEST' if test else filename  # Test purpose
        path_to_model = pjoin(path, filename)
        xgb_model.save_model(path_to_model)

    return xgb_model, path_to_model


def plot_prob_threshold(xgb_model, splits, config):
    """### Plot Probabilitiy Threshold """
    X_train, y_train, _, _, _, _ = splits
    # take min of (NUM_SAMPLES, length of unique probs in data)
    num_samples = min(config.plot_thresh_num_samples, len(np.unique(xgb_model.predict_proba(X_train)[:, 1])))
    ########-------------__#-#_#_#_#_#_#_#_#

    threshold = []
    accuracy = []
    precision = []
    recall = []
    f1_l = []
    neg_list = []

    for p in tqdm(np.random.choice(np.unique(xgb_model.predict_proba(X_train)[:, 1]), num_samples,
                                   replace=False)):  # iterate thresholds
        threshold.append(p)
        y_pred = (xgb_model.predict_proba(X_train)[:, 1] >= p).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

        neg_list.append(1 - (fn / (tn + fn)))  # Precision for opposite labeled data
        accuracy.append(balanced_accuracy_score(y_train, y_pred))
        prec = precision_score(y_train, y_pred)
        rec = recall_score(y_train, y_pred)
        f1 = 2 * prec * rec / (prec + rec)
        precision.append(1 - precision_score(y_train, y_pred))
        recall.append(recall_score(y_train, y_pred))
        f1_l.append(f1)

    plot_params = {
        "Balanced accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1_l,
        "Neg.Labeled Precision": neg_list
    }
    for label, metric_list in plot_params.items():
        plt.scatter(threshold, metric_list)
        plt.xlabel("Threshold")
        plt.ylabel(label)
        plt.show()
        print(f"Best Threshold in {label}:", threshold[np.argmax(accuracy)])


def plot_metrics_and_importance(xgb_model, df_before_process, splits, config, plot_tree=False, plot_auc_curve=True):
    X_train, y_train, _, _, X_clean_test, y_clean_test = splits

    """### Plot Importance and Metrics Plot"""

    # Get Features names before preprocess
    xgb_model.get_booster().feature_names = list(
        df_before_process.drop(columns=[config.label_column_name, config.bets_player_id_column]).columns.values)

    # Plot Feature-Importance plot
    xgb.plot_importance(xgb_model)

    # Plot Metrics
    results = xgb_model.evals_result()
    epochs = len(results['validation_0'][list(results['validation_0'].keys())[0]])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    metrics_to_plot = list(results['validation_0'].keys())
    for row in range(3):
        for col in range(3):
            if not metrics_to_plot:
                break
            ax[row, col].plot(x_axis, results['validation_0'][metrics_to_plot[0]], label='Train')
            ax[row, col].plot(x_axis, results['validation_1'][metrics_to_plot[0]], label='Val')
            ax[row, col].legend()
            ax[row, col].set_ylabel(f"{metrics_to_plot[0]}".capitalize())
            ax[row, col].title.set_text(f'XGBoost {metrics_to_plot[0]}')
            metrics_to_plot = metrics_to_plot[1:]

    plt.show()

    """Plot AUC Curve"""
    if plot_auc_curve:
        # Plot AUC curve
        # make predictions
        print("Results on Clean Data:")
        preds = xgb_model.predict(X_clean_test)
        tn, fp, fn, tp = confusion_matrix(y_clean_test, preds).ravel()
        print("TN", tn, "FP", fp, "FN", fn, "TP", tp)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        print(f'Precision:{prec:.2f}', f'Recall: {rec:.2f}', f'F1: {2 * prec * rec / (prec + rec)}',
              f'Neg. Precision: {1 - (fn / (tn + fn))}')

        fpr, tpr, thresholds = roc_curve(y_clean_test, preds)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve on Clean Test Data')
        plt.legend(loc="lower right")
        plt.show()

    """### Plot Tree"""
    if plot_tree:
        ##set up the parameters
        rcParams['figure.figsize'] = 200, 80
        # xgb.plot_importance(xgb_model)
        xgb.plot_tree(xgb_model, figsize=(200, 80))
        plt.show()
        rcParams['figure.figsize'] = 20, 10


def parameter_search(config, splits, search_type='grid'):
    """### Parameter Search"""

    def report_best_scores(results, n_top=3):
        best_candidate = None
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            if i == 1:
                best_candidate = results['params'][candidates[0]]
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters:")
                pprint.pprint(results['params'][candidate])  # pretty print
                print("")
        return best_candidate

    X_train, y_train, _, _, _, _ = splits
    positive_ratio = np.unique(y_train, return_counts=True)[1][0] / np.unique(y_train, return_counts=True)[1][
        1]  # get positive scale_ratio

    if search_type == 'random':  # Random search CV
        """
        #### RandomCV parameter search
        """
        # RandomCV #####
        model_params = config.random_cv_xgb_params
        # Create random generators - uniform or integer generating
        for key, val in model_params.items():
            if isinstance(val, list):
                continue
            else:  # create random generators
                model_params[key] = uniform(val['start'], val['interval']) if isinstance(val['start'],
                                                                                         float) else randint(
                    val['start'], val['end'])

        random_cv_params = config.random_cv_params
        random_cv_params['estimator'] = xgb.XGBClassifier()
        random_cv_params['param_distributions'] = model_params
        search = RandomizedSearchCV(**random_cv_params)
        search.fit(X_train, y_train)
        best_params = report_best_scores(search.cv_results_, 1)
    #####
    else:
        """#### GridsearchCV"""

        # GridSearchCV #####
        model_params = config.gridsearch_xgb_params
        # create lin or logspace or static vars for gridsearch
        for key, val in model_params.items():
            if isinstance(val, list):  # static
                continue
            else:  # create lin or logspace
                model_params[key] = np.logspace(val['start'], val['end'], val['steps'], base=2.0) if val[
                    'log_scale'] else np.linspace(val['start'], val['end'], val['steps'])

        grid_params = config.random_cv_params
        grid_params['estimator'] = xgb.XGBClassifier()
        grid_params['param_grid'] = model_params

        search = GridSearchCV(**grid_params)
        search.fit(X_train, y_train)

        best_params = report_best_scores(search.cv_results_, 1)
        #####
    return best_params


def main(config_path, path_to_file=None, test=False, search_params=False, inference=False):
    """
    Gets a path to files_df_dict with 'df_for_model' (and 'df_for_model_test' when not inference), train-val-test splits
    and train a XGBoost model.
    optional:
    - plot_prob_thresh: plot metrics for 50 (default) different Thresholds.
    - plot_metrics_and_importance: plot *training* and *val* metrics and feature-importance plot
    - search_params: Execute a RandomsearchCV or GridsearchCV for model parameters search
    :param config_path:
    :param path_to_file:
    :param test:
    :param search_params:
    :param inference:
    :return:
    """
    config = utils.load_config_from_path(config_path)
    parser = utils.parse_args_from_dict(config)

    if path_to_file:
        files_df_dict = utils.read_from_pickle(path_to_file,
                                               data_type=parser.sport_or_casino_task)  # TODO: Replace with read from DB
        if not inference:
            # Train-val-test split
            X, X_test = files_df_dict['df_for_model'], files_df_dict['df_for_model_test']
            splits = train_val_test_split(X, X_test, parser)

            model, path_to_model = train_model(splits, parser, test=test, save_model=True, model_filename='xgb_model')
            files_df_dict['trained_model'] = model
            if parser.plot_prob_thresh:
                plot_prob_threshold(model, splits, parser)
            if parser.plot_metrics_and_importance:
                plot_metrics_and_importance(model, df_before_process=X, splits=splits,
                                            config=parser, plot_tree=False, plot_auc_curve=True)
            if search_params:
                search_type = parser.search_params_type
                best_params = parameter_search(parser, splits, search_type)
                train_model(splits, parser, test=test, save_model=True, model_filename='xgb_model_paramsearch',
                            manual_model_params=best_params)
                files_df_dict['xgb_search_params'] = best_params
                files_df_dict['search_trained_model'] = model

        path = utils.dump_to_pickle(obj=files_df_dict,
                                    filename='block_3_train_model',
                                    data_type=parser.sport_or_casino_task) if not test \
            else utils.dump_to_pickle(obj=files_df_dict,
                                      filename='test_block_3_train_model',
                                      data_type=parser.sport_or_casino_task)
        return path
    else:
        raise Exception("Error in Path to pickle; Block_3")


if __name__ == '__main__':
    main()
