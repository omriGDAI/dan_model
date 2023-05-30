"""# Merge"""
import code_const
import argparse
from os.path import join as pjoin
from tqdm.auto import tqdm
import multiprocessing
import pandas as pd
import numpy as np

import read_data
import utils

tqdm.pandas()


def merge_tables(files_df_dict, config):
    """
    Merge columns from customers_df and transactions_df to 'Sports' and 'Casino' tables (inplace)
    (specified in config.filenames_to_merge_from) according to config.cols_to_merge_per_filename
    :param files_df_dict: dictionary of DFs concatenated
    :param config: configuration ArgParser
    :return: files_df_dict: modified dictionary of DFs concatenated
    """
    for filename in config.filenames_to_merge_from:
        for col in config.cols_to_merge_per_filename[filename]:
            print(f"Merging {col}..")
            for df_to_merge_to in config.dataframes_to_merge_to:
                mapping = dict(files_df_dict[filename][[config.sort_by_col, col]].values)
                files_df_dict[df_to_merge_to][col] = files_df_dict[df_to_merge_to][config.main_df_col_to_map_by].map(
                    mapping)
                print(
                    f"{df_to_merge_to}: Num of N/A rows after merge: {files_df_dict[df_to_merge_to][files_df_dict[df_to_merge_to][col].isna()].shape[0]}/{files_df_dict[df_to_merge_to].shape[0]}")

        # Drop N/A
        for df_to_merge_to in config.dataframes_to_merge_to:
            print(files_df_dict[df_to_merge_to][files_df_dict[df_to_merge_to][col].isna()])
            # if input("Drop N/A Rows?") in ['Y', 'y', 'yes', 'Yes', 'YES']:
            files_df_dict[df_to_merge_to].dropna(subset=config.cols_to_merge_per_filename[filename], inplace=True)
            print(f"Dropped N/A Rows. Total Rows Now: {files_df_dict[df_to_merge_to].shape[0]:,.0f}")

    return files_df_dict


def add_label_col(files_df_dict, config):
    """
    Adds label column to 'Casino' or 'Sports' table (according to config.sport_or_casino_task):
    - Filter transactions_df  (only 'approved deposits')
    - Aggregate transactions_df with 'sum' or 'num' of deposits (depends on the config.learn_task_vip flag)
    - Label 'active' users '1' or '0' according to config.active_user_label_1
    - Adds label column to data
    :param files_df_dict: dictionary of DFs concatenated
    :param config: configuration ArgParser
    :return: files_df_dict : dictionary of DFs concatenated
             active_users_df: summarized users with label - NOT IN USE
    """
    ##### ***Create Active Users DataFrame***
    condition = (files_df_dict[config.transactions_df_name][
                     config.deposit_withdraw_column_name] == config.deposit_name_in_column)
    condition &= (files_df_dict[config.transactions_df_name][
                      config.transaction_status_column_name] == config.trans_approved_in_column)
    data_df = files_df_dict[config.transactions_df_name][condition]
    # Active Users
    df = data_df.groupby(config.trans_player_id_column, as_index=False).agg(
        {config.trans_id_column: 'count'}).sort_values(config.trans_id_column,
                                                       ascending=True)  # count each user

    if config.learn_task_vip:  # VIP Task
        df = data_df.groupby(config.trans_player_id_column, as_index=False).agg(
            {config.trans_amount_col: 'sum'}).sort_values(config.trans_amount_col,
                                                          ascending=True)  # sum for each user

        active_users_df = df[
            df[config.trans_amount_col] > config.active_users_sum_deposits_vip]

    else:  # STD Task
        active_users_df = df[
            df[config.trans_id_column] > config.active_users_num_deposits]

    print('All Users')
    print(df.describe())

    print()
    print(active_users_df.nunique())

    """##### Add Active user column"""

    data_df = files_df_dict[config.sport_or_casino_task]

    good_keys = list(set(data_df.player_id) & set(active_users_df[config.trans_player_id_column]))
    print('Length of player_id intersection of transactions and bets:', len(good_keys))

    data_df[config.label_column_name] = True if not config.active_user_label_1 else False
    data_df.loc[data_df[config.bets_player_id_column].isin(
        good_keys), config.label_column_name] = 'False' if not config.active_user_label_1 else 'True'
    print(data_df[config.label_column_name].value_counts())

    # if input("Drop N/A Rows?") in ['Y','y','yes','Yes','YES']:
    data_df.dropna(subset=[config.label_column_name], inplace=True)
    print(f"Dropped N/A Rows. Total Rows Now: {data_df.shape[0]:,.0f}")

    return files_df_dict, active_users_df


def add_agg_columns(files_df_dict, config, multiproc=False):
    """##### Add Agg Columns to 'Casino' or 'Sports' table (according to config.sport_or_casino_task):
    - Add last_transaction date and last_transaction amount column using add_last_transaction function
    - Add sum_deposits and num of deposits columns using add_sum_num_deposits function
    - Map the generated agg. columns to the 'Casino' or 'Sports' table using map_agg_cols_to_main_df
    :param files_df_dict: dictionary of DFs concatenated
    :param config: configuration ArgParser
    :param multiproc: Multiprocess BOOL - NOT IN USE
    :return: files_df_dict : dictionary of DFs concatenated
    """
    """###### Prepare Transactions DF and grouped DF for usage"""

    """ Local Variables Definition"""
    # Prepare transactions
    condition = (files_df_dict[config.transactions_df_name][config.trans_player_id_column].isin(
        list(set(files_df_dict[config.sport_or_casino_task].player_id))))
    condition &= (files_df_dict[config.transactions_df_name][
                      config.deposit_withdraw_column_name] == config.deposit_name_in_column)
    condition &= (files_df_dict[config.transactions_df_name][
                      config.transaction_status_column_name] == config.trans_approved_in_column)
    transaction_date_series = files_df_dict[config.transactions_df_name][condition]
    good_keys = list(set(files_df_dict[config.sport_or_casino_task].player_id) & set(
        transaction_date_series[config.trans_player_id_column]))
    transaction_date_series = files_df_dict[config.transactions_df_name][condition].set_index(
        [config.trans_player_id_column])

    transaction_date_series.loc[:, config.trans_date_column_name] = pd.to_datetime(
        transaction_date_series.loc[:, config.trans_date_column_name])
    transaction_date_series.reset_index(inplace=True)
    df_grouped_member = transaction_date_series.groupby([config.trans_player_id_column])[config.trans_date_column_name]
    df_grouped_member_all_cols = transaction_date_series.groupby([config.trans_player_id_column])

    # prepare players_n_bets df
    players_n_bet_dates = files_df_dict[config.sport_or_casino_task][
        files_df_dict[config.sport_or_casino_task][config.bets_player_id_column].isin(good_keys)][
        [config.bet_id_column, config.bets_player_id_column, config.bet_date_column]]
    players_n_bet_dates.loc[:, config.bet_date_column] = pd.to_datetime(
        players_n_bet_dates.loc[:, config.bet_date_column])

    """###### Utils"""

    def nearest(items, pivot):
        return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot), default=np.nan)

    def earliest(items, pivot):
        return max([i for i in items if i <= pivot], key=lambda x: abs(x - pivot), default=np.nan)

    def last_transaction_n_amount_func(x):
        nearest_date = nearest(df_grouped_member.get_group(x[config.bets_player_id_column]), x[config.bet_date_column])
        amount_deposit = df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
            df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
                config.trans_date_column_name] == nearest_date][config.trans_amount_col]
        if amount_deposit.size != 0:
            amount_deposit = amount_deposit.values[0]
        return nearest_date, amount_deposit

    def sum_deposit_func(x):
        earliest_date = earliest(df_grouped_member.get_group(x[config.bets_player_id_column]),
                                 x[config.bet_date_column])
        init_amount_deposit = df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
            df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
                config.trans_date_column_name] == earliest_date][config.trans_amount_col]
        if init_amount_deposit.size != 0:  # verify it's not nan
            init_amount_deposit = init_amount_deposit.values[0]
            # earliest_date =  earliest(df_grouped_member.get_group(x[config.bets_player_id_column]),x[config.bet_date_column])

            condition = df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
                            config.trans_date_column_name] >= earliest_date
            condition &= df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[
                             config.trans_date_column_name] <= x[config.bet_date_column]

            df_filtered = df_grouped_member_all_cols.get_group(x[config.bets_player_id_column])[condition][
                config.trans_amount_col]
            sum_deposit, num_deposit = df_filtered.sum(), df_filtered.shape[0]
        else:
            sum_deposit = np.nan
            num_deposit = np.nan
        return sum_deposit, num_deposit

    def _apply_df(args):
        df, func, num, kwargs = args
        q = kwargs.pop('queue')
        s = kwargs.pop('sema')
        q.put(df.apply(func, **kwargs))
        s.release()
        # kwargs['queue']=q
        # return num, df.apply(func, **kwargs)

    def apply_by_multiprocessing(df, func, num_splits=8, **kwargs):
        workers = kwargs.pop('workers')
        processes_list, df_list = [], []
        queue = multiprocessing.Queue()
        sema = multiprocessing.Semaphore(8)
        kwargs['queue'] = queue
        kwargs['sema'] = sema
        args_list = [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers * num_splits))]
        for splits_chunk_ind in tqdm(range(num_splits)):
            for w_ind in range(workers):
                sema.acquire()
                process = multiprocessing.Process(target=_apply_df,
                                                  args=(args_list[w_ind + splits_chunk_ind * workers],))
                process.daemon = True
                processes_list.append(process)
                process.start()
            for p in processes_list:
                p.join()
                # p.close()
                # processes_list.remove(p)
                # p.kill()

        while not queue.empty():
            df_list.append(queue.get())

        # pool = multiprocessing.Pool(processes=workers)
        # result = tqdm(pool.imap(_apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers*num_splits))]), total=len([(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers*num_splits))]))
        # pool.close()

        # result = sorted(result,key=lambda x:x[0])
        return pd.concat([i for i in df_list])

    def _save_partial_players_bets(df, config, filename_addition='last_transaction'):
        filename = f'players_n_bet_dates_{filename_addition}.csv'
        path = pjoin(config.DATA_FOLDER_PATH, config.DATA_DIR, filename)
        df.to_csv(path)
        print("Path Saved is:", path)

    def load_partitioned_players_bets(folder_path=None):
        if folder_path:
            df = [
                pd.read_csv(pjoin(folder_path, f'players_n_bet_dates_last_transaction_part_{ind}.csv'), index_col=False)
                for ind in range(10)]
        else:
            df = [pd.read_csv(
                pjoin(config.DATA_FOLDER_PATH, config.DATA_DIR, f'players_n_bet_dates_last_transaction_part_{ind}.csv'),
                index_col=False) for ind in range(10)]
        return pd.concat(df)

    def load_merged_players_bets(folder_path=None):
        if folder_path:
            df = pd.read_csv(
                pjoin(folder_path, 'players_n_bet_dates_last_transaction_and_amount_and_sum_and_counts.csv'),
                index_col=False)
        else:
            df = pd.read_csv(pjoin(config.DATA_FOLDER_PATH, config.DATA_DIR,
                                   'players_n_bet_dates_last_transaction_and_amount_and_sum_and_counts.csv'),
                             index_col=False)
        return df

    """###### Last Transaction *apply*"""

    def add_last_transaction(players_n_bet_dates_df, save_flag=False, workers=2, use_mutliproc=False):
        if use_mutliproc:
            pool = multiprocessing.Pool()
            with tqdm(total=len(players_n_bet_dates_df)) as pbar:
                results = pool.imap_unordered(last_transaction_n_amount_func,
                                              players_n_bet_dates_df)  # use imap to get pbar updated
            pool.close()
            players_n_bet_dates_df['last_transaction'] = results[0]
            players_n_bet_dates_df['last_deposit_amount'] = results[1]

            # players_n_bet_dates_df[['last_transaction', 'last_deposit_amount']] = apply_by_multiprocessing(players_n_bet_dates_df,last_transaction_n_amount_func, axis=1,result_type='expand',
            #                                                       workers=workers)

        else:
            players_n_bet_dates_df[['last_transaction', 'last_deposit_amount']] = players_n_bet_dates_df.progress_apply(
                last_transaction_n_amount_func, axis=1, result_type='expand')
        ## run by 4 processors
        if save_flag:  # TODO: implement save
            _save_partial_players_bets(filename_addition='last_transaction')
        return players_n_bet_dates_df

    def add_sum_num_deposits(players_n_bet_dates_df, save_flag=False, workers=2, use_mutliproc=False):
        if use_mutliproc:
            pool = multiprocessing.Pool()
            with tqdm(total=len(players_n_bet_dates_df)) as pbar:
                results = pool.imap_unordered(sum_deposit_func, players_n_bet_dates_df)  # use imap to get pbar updated
            pool.close()
            players_n_bet_dates_df[config.sum_deposit_col_name] = results[0]
            players_n_bet_dates_df[config.num_deposit_col_name] = results[1]

            # players_n_bet_dates_df[[config.sum_deposit_col_name, config.num_deposit_col_name]] = apply_by_multiprocessing(players_n_bet_dates_df,sum_deposit_func, axis=1,result_type='expand',
            #                                                       workers=workers)
        else:
            players_n_bet_dates_df[
                [config.sum_deposit_col_name, config.num_deposit_col_name]] = players_n_bet_dates_df.progress_apply(
                sum_deposit_func, axis=1,
                result_type='expand')
        ## run by 4 processors
        if save_flag:  # TODO: implement save
            _save_partial_players_bets(filename_addition='last_transaction')
        return players_n_bet_dates_df

    """###### Set back to the main DF"""

    def map_agg_cols_to_main_df(files_df_dict, players_n_bet_dates_df, config, sample_test=False):
        # if sample_test - JUST FOR TEST - Samples 1% of the data
        ##### --------------------------------- ######

        ids_in_players_n_bets = files_df_dict[config.sport_or_casino_task][config.bet_id_column].isin(
            players_n_bet_dates_df.sample(frac=0.01, random_state=42)[config.bet_id_column]) if sample_test else \
            files_df_dict[config.sport_or_casino_task][config.bet_id_column].isin(
                players_n_bet_dates_df[config.bet_id_column])  # also the same order
        cols = ['last_transaction', 'last_deposit_amount', config.sum_deposit_col_name, config.num_deposit_col_name]
        files_df_dict[config.sport_or_casino_task].loc[ids_in_players_n_bets, cols] = \
            players_n_bet_dates_df.sample(frac=0.01, random_state=42)[cols] if sample_test else players_n_bet_dates_df[
                cols]
        # files_df_dict[config.sport_or_casino_task].rename(columns={"last_bet": "last_transaction"},
        #                                            inplace=True)  # code compatibility later on
        return files_df_dict

    """"#### add_agg_columns - MAIN ##"""
    # add agg columns to players_n_bet_dates df
    print("Adding Last Deposit transaction and last amount...")
    players_n_bet_dates = add_last_transaction(players_n_bet_dates, workers=2, use_mutliproc=multiproc)
    print("Adding Num and Sum of deposits...")
    players_n_bet_dates = add_sum_num_deposits(players_n_bet_dates, workers=2, use_mutliproc=multiproc)

    # add players_n_bet_dates_df columns to file_df_dict
    files_df_dict = map_agg_cols_to_main_df(files_df_dict, players_n_bet_dates, config)

    return files_df_dict


def main(config_path, path_to_file=None, test=False, test_frac=0.01, inference=False):
    config = utils.load_config_from_path(config_path)
    parser = utils.parse_args_from_dict(config)

    if path_to_file:
        files_df_dict = utils.read_from_pickle(path_to_file,
                                               data_type=parser.sport_or_casino_task)  # TODO: Replace with read from DB

        # Test on 1% of Data
        files_df_dict[parser.sport_or_casino_task] = files_df_dict[parser.sport_or_casino_task].sample(frac=test_frac,
                                                                                                       random_state=42) \
            if test else files_df_dict[parser.sport_or_casino_task]  # TODO: Remove

        merge_tables(files_df_dict, parser)  # merge columns from other table

        if not inference:  # in inference-flow, data has no labels
            add_label_col(files_df_dict, parser)

        files_df_dict = add_agg_columns(files_df_dict, parser, multiproc=False)

        path = utils.dump_to_pickle(obj=files_df_dict,
                                    filename='block_1_generate_complete_df',
                                    data_type=parser.sport_or_casino_task) if not test \
            else utils.dump_to_pickle(obj=files_df_dict,
                                      filename='test_block_1_generate_complete_df',
                                      data_type=parser.sport_or_casino_task)

        return path
    else:
        raise Exception("Error in Path to pickle; Block_1")
#
# if __name__ == '__main__':
#     # TODO: add description for each function in main
#     # TODO: add read from db/ local data - and comment for omri
#     # TODO: multiprocess not working in local macbook
#     multiprocessing.set_start_method('fork')
#     main()
# print("Done")
