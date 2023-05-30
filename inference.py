import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import utils


def model_predict(xgb_model, df_dict, config):
    """predict without checking performance"""
    try:
        assert config.label_column_name not in df_dict['df_for_model'].columns, "Has Label Column in *Inference* Flow"
        X = df_dict['df_for_model'].drop(columns=[config.bets_player_id_column]).values.astype(float)
    except:
        if input("Label_Column is in Table Columns. \nContinue?") in 'Yy':
            X = df_dict['df_for_model'].drop(
                columns=[config.bets_player_id_column, config.label_column_name]).values.astype(float)
            pass
    # X = df_dict['df_for_model'].drop(columns=[config.bets_player_id_column]).values.astype(float)
    # make predictions
    xgb_params = config.xgb_params
    pprint.pprint(xgb_params)  # pretty print
    print("Task is VIP") if config.learn_task_vip else print("Task is STD")
    print(f"Active Users are Labeled as '{int(config.active_user_label_1)}'")
    preds = xgb_model.predict(X)
    X_with_preds = df_dict['df_for_model'].copy()
    X_with_preds[config.label_column_name] = preds
    df_dict['df_with_predictions'] = X_with_preds


def test_results(model, df_dict, config, has_search_params=False):
    """predict and check performance"""

    data = df_dict['df_for_model_test']
    # make predictions
    X_test, y_test = data.drop(columns=[config.bets_player_id_column, config.label_column_name]).values.astype(float), \
        data[config.label_column_name].values.astype(float)
    xgb_params = config.xgb_params if not has_search_params else df_dict['xgb_search_params']
    pprint.pprint(xgb_params)  # pretty print
    print("Task is VIP") if config.learn_task_vip else print("Task is STD")
    print(f"Active Users are Labeled as '{int(config.active_user_label_1)}'")
    print("Results on Clean Test Data:")
    preds = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("TN", tn, "FP", fp, "FN", fn, "TP", tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f'Precision:{prec:.2f}', f'Recall: {rec:.2f}', f'F1: {2 * prec * rec / (prec + rec)}',
          f'Neg. Precision: {1 - (fn / (tn + fn))}')


def post_process_results(model, df_dict, config, inference=False, full_process=True):
    def get_preds_df_per_user(model, data, config, y_clean_test=None, inference=None):
        if inference:
            try:
                assert config.label_column_name not in df_dict[
                    'df_for_model'].columns, "Has Label Column in *Inference* Flow"
                drop_cols = [config.bets_player_id_column]
            except:
                if input("Label_Column is in Table Columns. \nContinue?") in 'Yy':
                    drop_cols = [config.bets_player_id_column, config.label_column_name]
                    has_label_column = True
                    pass
        else:
            drop_cols = [config.bets_player_id_column, config.label_column_name]
        threshold = config.model_best_threshold

        preds = (model.predict_proba(data.drop(columns=drop_cols).values.astype(float))[:, 1] >= threshold).astype(int)

        df_preds = pd.concat(
            [data[[config.bet_date_column, config.bets_player_id_column, config.sum_deposit_col_name,
                   config.num_deposit_col_name]].reset_index(drop=True),
             pd.Series(preds, name='prediction')], axis=1)
        df_preds = pd.concat([df_preds, y_clean_test.reset_index(drop=True)],
                             axis=1) if y_clean_test is not None else df_preds  # if
        df_preds.sort_values(config.sum_deposit_col_name, ascending=False)

        if y_clean_test is not None:  # if testing and not in inference
            if config.learn_task_vip:
                df_preds[df_preds[config.sum_deposit_col_name] < config.active_users_sum_deposits_vip][
                    [config.label_column_name, 'prediction']].value_counts()
            else:
                df_preds[[config.label_column_name, 'prediction']].value_counts()

        return df_preds

    X = df_dict['df_for_model_test'] if not inference else df_dict['df_for_model']

    if not inference:
        X_test, y_test = X, X[config.label_column_name]
        df_predictions = get_preds_df_per_user(model, X_test, y_clean_test=y_test, config=config, inference=inference)

    else:
        df_predictions = get_preds_df_per_user(model, X, config=config, inference=inference)

    """Voting Function"""

    df_group_id = df_predictions.groupby(config.bets_player_id_column)

    # Voting per user input
    def voting_func_user(x, voting_thr=config.voting_func_threshold):
        df = df_group_id.get_group(x)['prediction']
        ratio = df.sum() / df.count() if df.count() > 0 else 0
        return 1 if ratio > config.voting_func_threshold else 0

    # Voting per DataFrame
    def voting_func_df(x, voting_thr=config.voting_func_threshold):
        ratio = x['prediction'].sum() / x['prediction'].count() if x['prediction'].count() > 0 else 0
        return 1 if ratio > config.voting_func_threshold else 0

    # Last prediction check
    # Populate per user prediction before reaching threshold for VIP or num_deposits

    last_ratio = 0.5  # top ratio of sum_deposit, to check before giving prediction

    df_predictions.loc[:, 'voting_pred_thresh'] = -1
    for user in df_predictions.player_id.unique():
        condition = df_predictions.player_id == user
        if config.learn_task_vip:
            condition &= df_predictions[
                             config.sum_deposit_col_name] <= last_ratio * config.active_users_sum_deposits_vip
        else:
            condition &= df_predictions[config.num_deposit_col_name] <= config.active_users_num_deposits
        df_predictions.loc[condition, 'voting_pred_thresh'] = voting_func_user(user,
                                                                               voting_thr=config.voting_func_threshold)

    # Remove not causal rows
    df_causual_predictions = df_predictions[df_predictions['voting_pred_thresh'] >= 0]

    if not inference:  # if testing and not in inference

        df_causual_predictions['last_prediction_agree_gt_and_active'] = (
                                                                                df_causual_predictions.voting_pred_thresh == df_causual_predictions.active_user) & (
                                                                                df_causual_predictions.voting_pred_thresh == int(
                                                                            config.active_user_label_1))
        df_causual_predictions[
            'last_prediction_agree_gt'] = df_causual_predictions.voting_pred_thresh == df_causual_predictions.active_user
        correct = df_causual_predictions.sort_values(config.bet_date_column).drop_duplicates(
            config.bets_player_id_column,
            keep='last').last_prediction_agree_gt.sum()
        correct_n_active = df_causual_predictions.sort_values(config.bet_date_column).drop_duplicates(
            config.bets_player_id_column,
            keep='last').last_prediction_agree_gt_and_active.sum()

        print(
            f"Results based on data up to {last_ratio} of VIP Sum Defined ({config.active_users_sum_deposits_vip})") if config.learn_task_vip else print(
            f"Results based on Data up to Second Deposit")
        print("Num of Users Correct:", correct, "; Out of Total:", df_causual_predictions.player_id.nunique(),
              f'; %{100 * (correct / df_causual_predictions.player_id.nunique()):.2f}')
        print("Num of Users Correct Out of Active Predicted:", correct_n_active, "; Out of Total:",
              df_causual_predictions[
                  df_causual_predictions.voting_pred_thresh == int(config.active_user_label_1)].player_id.nunique(),
              f'; %{100 * (correct_n_active / df_causual_predictions[df_causual_predictions.voting_pred_thresh == int(config.active_user_label_1)].player_id.nunique()):.2f}')

    """Populate "Rolling Voting Prediction"
    """
    if full_process:
        def populate_voting_pred(x, vote_func=voting_func_df):
            condition = df_group_id.get_group(x[config.bets_player_id_column])[config.bet_date_column] <= x[
                config.bet_date_column]
            if config.learn_task_vip:
                condition &= df_group_id.get_group(x[config.bets_player_id_column])[
                                 config.sum_deposit_col_name] <= config.active_users_sum_deposits_vip
            else:
                condition &= df_group_id.get_group(x[config.bets_player_id_column])[
                                 config.num_deposit_col_name] <= config.active_users_num_deposits
            df_filtered = df_group_id.get_group(x[config.bets_player_id_column])[condition]
            return vote_func(df_filtered, voting_thr=config.voting_func_threshold)

        voting_pred_series = df_predictions.progress_apply(populate_voting_pred, axis=1)
        # voting_pred_series = apply_by_multiprocessing(df_predictions, populate_voting_pred, axis=1, workers=4)
        df_predictions['votin_pred'] = voting_pred_series

    return df_predictions


def test_results_on_users(preds_df, config, score_func=np.average, vip_ratio=1.0, return_score_df=False,
                          inference=False):
    """Calculate Mean Score over users"""

    if config.learn_task_vip:
        condition = preds_df[config.sum_deposit_col_name] < vip_ratio * config.active_users_sum_deposits_vip
    else:
        condition = preds_df[config.num_deposit_col_name] <= config.active_users_num_deposits

    df_causual_predictions = preds_df[condition]
    df_causual_predictions[
        'prediction_agree_gt'] = df_causual_predictions.votin_pred == df_causual_predictions.active_user
    agree_condition = (df_causual_predictions.votin_pred == df_causual_predictions.active_user)
    agree_condition &= (df_causual_predictions.votin_pred == int(config.active_user_label_1))
    df_causual_predictions['prediction_agree_gt_and_active'] = agree_condition

    user_accuracy_list = []
    user_precision_list = []
    for user in df_causual_predictions.player_id.unique():
        condition = df_causual_predictions.player_id == user
        if config.learn_task_vip:
            condition &= df_causual_predictions[config.sum_deposit_col_name] <= config.active_users_sum_deposits_vip
        else:
            condition &= df_causual_predictions[config.num_deposit_col_name] <= config.active_users_num_deposits
        correct = df_causual_predictions[condition].prediction_agree_gt.sum()
        correct_n_active = df_causual_predictions[condition].prediction_agree_gt_and_active.sum()

        user_accuracy_list.append(correct / df_causual_predictions[condition].shape[0])
        condition &= (df_causual_predictions.votin_pred == int(config.active_user_label_1))
        if df_causual_predictions[condition].shape[0]:
            user_precision_list.append(correct_n_active / df_causual_predictions[condition].shape[0])

    score = score_func(user_accuracy_list)

    print(f"Score on all users: %{100 * score_func(user_accuracy_list):.2f}")
    print(f"Score on Active users - (i.e. Model Predicted Active): %{100 * score_func(user_precision_list):.2f}")
    return df_causual_predictions if return_score_df else None


def main(config_path, path_to_file=None, inference_flow=False, full_post_process=True,
         test=True, path_to_model=None, search_params=False):
    config = utils.load_config_from_path(config_path)
    parser = utils.parse_args_from_dict(config)

    if path_to_file:
        files_df_dict = utils.read_from_pickle(path_to_file,
                                               data_type=parser.sport_or_casino_task)  # TODO: Replace with read from DB
        # Read Data, Read Model
        model = files_df_dict['trained_model'] if not inference_flow else utils.read_model(path_to_model)
        if not inference_flow:
            test_results(model, files_df_dict, parser)
        else:
            model_predict(model, files_df_dict, parser)
        if search_params:  # only in training flow
            print("Param Search Model Results:")
            searched_model = files_df_dict['search_trained_model']
            test_results(searched_model, files_df_dict, parser, has_search_params=search_params)

        df_preds = post_process_results(model, files_df_dict, parser, inference=inference_flow,
                                        full_process=full_post_process)  # convert to users and voting decision
        if not inference_flow:
            if full_post_process:
                test_results_on_users(df_preds, parser, vip_ratio=0.5,
                                      inference=inference_flow)  # test_results_on_users(df_preds, vip_ratio=0.5) for POC
        if search_params:
            df_preds = post_process_results(searched_model, files_df_dict, parser, inference=inference_flow,
                                            full_process=full_post_process)  # convert to users and voting decision
            if full_post_process:
                test_results_on_users(df_preds, parser, vip_ratio=0.5,
                                      inference=inference_flow)  # test_results_on_users(df_preds, vip_ratio=0.5) for POC

        path = utils.dump_to_pickle(obj=files_df_dict,
                                    filename='block_4_inference',
                                    data_type=parser.sport_or_casino_task) if not test \
            else utils.dump_to_pickle(obj=files_df_dict,
                                      filename='test_block_4_inference',
                                      data_type=parser.sport_or_casino_task)
        return path
    else:
        raise Exception("Error in Path to pickle; Block_4")


if __name__ == '__main__':
    main()
