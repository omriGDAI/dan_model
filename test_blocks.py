import json
import os
from os.path import join as pjoin

import code_const

# TODO: OMRI - GO OVER ARGS AND SAY ITS OK

"""Block 0"""
config_0 = dict(
    data_source=code_const.DATA_SOURCE,
    data_dir=code_const.DATA_DIR,
    data_folder_path=code_const.DATA_FOLDER_PATH,
    csv_encoding=code_const.CSV_ENCODING,
    unzip_flag=code_const.UNZIP_FLAG,
    remove_extracted=code_const.REMOVE_EXTRACTED,
    files_to_concat=code_const.FILES_TO_CONCAT,
    main_df_to_merge=code_const.MAIN_DF_TO_MERGE,
    sport_or_casino_task=code_const.SPORT_OR_CASINO_TASK,
)

"""Block 1"""
config_1 = dict(
    dataframes_to_merge_to=code_const.DATAFRAMES_TO_MERGE_TO,
    sort_by_col=code_const.SORT_BY_COL,
    main_df_col_to_map_by=code_const.MAIN_DF_COL_TO_MAP_BY,
    filenames_to_merge_from=code_const.FILENAMES_TO_MERGE_FROM,
    cols_to_merge_per_filename=code_const.COLS_TO_MERGE_PER_FILENAME,
    transactions_df_name=code_const.TRANSACTIONS_DF_NAME,
    sport_or_casino_task=code_const.SPORT_OR_CASINO_TASK,
    deposit_withdraw_column_name=code_const.DEPOSIT_WITHDRAW_COLUMN_NAME,
    deposit_name_in_column=code_const.DEPOSIT_NAME_IN_COLUMN,
    transaction_status_column_name=code_const.TRANSACTION_STATUS_COLUMN_NAME,
    trans_approved_in_column=code_const.TRANS_APPROVED_IN_COLUMN,
    trans_player_id_column=code_const.TRANS_PLAYER_ID_COLUMN,
    trans_id_column=code_const.TRANS_ID_COLUMN,
    trans_amount_col=code_const.TRANS_AMOUNT_COL,
    trans_date_column_name=code_const.TRANS_DATE_COLUMN_NAME,
    label_column_name=code_const.LABEL_COLUMN_NAME,
    bet_id_column=code_const.BET_ID_COLUMN,
    bets_player_id_column=code_const.BETS_PLAYER_ID_COLUMN,
    bet_date_column=code_const.BET_DATE_COLUMN,
    sum_deposit_col_name=code_const.SUM_DEPOSIT_COLUMN_NAME,
    num_deposit_col_name=code_const.NUM_DEPOSITS_COLUMN_NAME,
    # Training Task params
    active_users_num_deposits=code_const.ACTIVE_NUM_DEPOSITS,
    active_users_sum_deposits_vip=code_const.VIP_SUM_DEPOSITS,
    active_user_label_1=code_const.ActiveUser_LABEL_is_1,
    user_ratio_in_test=code_const.RATIO_USER_IN_TEST,
    learn_task_vip=code_const.TaskVIP,
)

"""Block 2"""
config_2 = dict(
    data_dir=code_const.DATA_DIR,
    data_folder_path=code_const.DATA_FOLDER_PATH,
    transactions_df_name=code_const.TRANSACTIONS_DF_NAME,
    trans_amount_col=code_const.TRANS_AMOUNT_COL,
    trans_id_column=code_const.TRANS_ID_COLUMN,
    trans_player_id_column=code_const.TRANS_PLAYER_ID_COLUMN,
    preprocess_params=code_const.preprocess_params,
    sport_or_casino_task=code_const.SPORT_OR_CASINO_TASK,
    active_users_num_deposits=code_const.ACTIVE_NUM_DEPOSITS,
    active_users_sum_deposits_vip=code_const.VIP_SUM_DEPOSITS,
    bets_player_id_column=code_const.BETS_PLAYER_ID_COLUMN,
    learn_task_vip=code_const.TaskVIP,
    user_ratio_in_test=code_const.RATIO_USER_IN_TEST,
    # If change_labels==True, need the parameters below
    deposit_withdraw_column_name=code_const.DEPOSIT_WITHDRAW_COLUMN_NAME,
    deposit_name_in_column=code_const.DEPOSIT_NAME_IN_COLUMN,
    transaction_status_column_name=code_const.TRANSACTION_STATUS_COLUMN_NAME,
    trans_approved_in_column=code_const.TRANS_APPROVED_IN_COLUMN,
    label_column_name=code_const.LABEL_COLUMN_NAME,
    active_user_label_1=code_const.ActiveUser_LABEL_is_1,
)

"""Block 3"""
config_3 = dict(
    data_dir=code_const.DATA_DIR,
    data_folder_path=code_const.DATA_FOLDER_PATH,
    plot_prob_thresh=code_const.PLOT_PROB_THRESH,
    plot_metrics_and_importance=code_const.PLOT_METRICS_AND_IMPORTANCE,
    search_params=code_const.SEARCH_FOR_PARAMS_FLAG,
    search_params_type=code_const.SEARCH_PARAMS_TYPE,
    bets_player_id_column=code_const.BETS_PLAYER_ID_COLUMN,
    user_ratio_in_val=code_const.VAL_PLAYERS_RATIO,
    label_column_name=code_const.LABEL_COLUMN_NAME,
    xgb_params=code_const.XGB_TRAIN_PARAMS,
    sport_or_casino_task=code_const.SPORT_OR_CASINO_TASK,
    learn_task_vip=code_const.TaskVIP,
    active_user_label_1=code_const.ActiveUser_LABEL_is_1,
    active_users_sum_deposits_vip=code_const.VIP_SUM_DEPOSITS,
    plot_thresh_num_samples=code_const.PLOT_THRESH_NUM_SAMPLES,
    random_cv_params=code_const.RANDOM_CV_META_PARAMS,
    random_cv_xgb_params=code_const.RANDOM_CV_XGB_PARAMS,
)

"""Block 4"""
config_4 = dict(
    bet_date_column=code_const.BET_DATE_COLUMN,
    bets_player_id_column=code_const.BETS_PLAYER_ID_COLUMN,
    sum_deposit_col_name=code_const.SUM_DEPOSIT_COLUMN_NAME,
    num_deposit_col_name=code_const.NUM_DEPOSITS_COLUMN_NAME,
    model_best_threshold=code_const.XGB_BEST_THRESHOLD,
    sport_or_casino_task=code_const.SPORT_OR_CASINO_TASK,
    learn_task_vip=code_const.TaskVIP,
    active_users_sum_deposits_vip=code_const.VIP_SUM_DEPOSITS,
    active_users_num_deposits=code_const.ACTIVE_NUM_DEPOSITS,
    active_user_label_1=code_const.ActiveUser_LABEL_is_1,
    label_column_name=code_const.LABEL_COLUMN_NAME,
    voting_func_threshold=code_const.VOTING_THR,
    xgb_params=code_const.XGB_TRAIN_PARAMS,
)


def generate_config_files(config_folder_path, data_type, learn_task_vip, test=False):
    path = pjoin(config_folder_path, data_type)
    if learn_task_vip:
        path = pjoin(path, f'VIP_{code_const.VIP_SUM_DEPOSITS}')
    else:
        path = pjoin(path, f'STD')

    if test:
        path = pjoin(path, 'TEST')

    if not os.path.exists(path):  # create folders
        os.makedirs(path)

    config_path_dict = {}
    for i, config in enumerate([config_0, config_1, config_2, config_3, config_4]):
        curr_path = pjoin(path, f'config_{i}.json')
        with open(curr_path, 'w') as fp:
            json.dump(config, fp)
        config_path_dict[f'config_{i}'] = curr_path

    return config_path_dict


def generate_data_paths(phase, data_type, test):
    filename_0 = 'block_0_read_data.pickle'
    filename_1 = 'block_1_generate_complete_df.pickle'
    filename_2 = 'block_2_prepare_data_for_model.pickle'
    filename_3 = 'block_3_train_model.pickle'
    path_dict = {}

    for i, filename in zip(range(4), [filename_0, filename_1, filename_2, filename_3]):
        path_dict[f'block_{i}'] = pjoin('Data', data_type, filename) if not test else pjoin('Data', data_type,
                                                                                            'test_' + filename)

    for ascend_ind, descend_ind in zip(list(range(4)), list(range(4))[::-1]):
        if phase == 0:
            return dict()
        if phase - 1 == descend_ind:
            break
        else:
            path_dict.pop(f'block_{descend_ind}')
    return path_dict
