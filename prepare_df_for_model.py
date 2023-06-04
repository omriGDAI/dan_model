"""
## Pre-process

### Definition
"""

import pandas as pd
import numpy as np
import utils
from sklearn import preprocessing


from generate_complete_df import add_label_col


def create_dataset_from_table(files_df_dict, config, inference=False, change_labels=False):
    """
    Create a dataset table from 'Casino' or 'Sports' table (according to config.sport_or_casino_task)
    - Remove bets after label has reached in data (i.e. reached sum of deposits for VIP or second deposit for STD)
    - preprocess_cols function :
        - map label col to '0', '1';
        - encode categorical columns;
        - datetime cols to timestamp and adds diff column between them (i.e. date1-date2 column)
        - add numerical transform columns - norm and lognorm columns
    - Sample users for Test
    :param files_df_dict: files_df_dict - dictionary of DFs
    :param config: configuration ArgParser
    :param inference: Bool - indicates if inference flow
    :param change_labels: Bool - if needs to change a data labels (e.g. change VIP sums definition) according to config
    :return: files_df_dict - dictionary of DFs- addition of 'df_for_model' (and 'df_for_model_test' if not inference flow)
    """

    def preprocess_cols(df,
                        drop_cols=[],
                        cat_to_num_cols=[],
                        target_col=None,
                        date_col=[],
                        numerical_to_transform_cols=[],
                        inference_flow=False,
                        ):
        df_data = df  # df.copy()
        if target_col is not None:
            df_data.loc[:, target_col] = df_data.loc[:, target_col].astype(str)
        label_enc_dict = {col: preprocessing.LabelEncoder() for col in cat_to_num_cols+[target_col]}
        cols_to_encode = cat_to_num_cols+[target_col] if not inference else cat_to_num_cols
        for col in cols_to_encode:
            df_data.loc[:, col] = label_enc_dict[col].fit_transform(df_data.loc[:, col])

        # Datetime to timestamp
        first = True
        for ind, col in enumerate(date_col):
            df_data.loc[:, col] = pd.to_datetime(df_data[col]).values.astype(np.int64) // 10 ** 9
            if first:
                temp = df_data.loc[:, col].copy()
                first = False
                continue
            else:
                temp = temp.subtract(df_data.loc[:, col])
                df_data[f'diff_{ind}'] = temp.copy()
                temp = df_data.loc[:, col].copy()

        def safe_log10(x, eps=1e-10):
            x = x.astype(float)
            result = np.where(x > eps, x, -10)
            np.log10(result, out=result, where=result > 0)
            return result

        for col in numerical_to_transform_cols:
            df_data.loc[:, col] = df_data.loc[:, col].astype(np.float64)
            df_data.loc[:, f'norm_{col}'] = 2 * (df_data[col] - df_data[col].min()) / (
                        df_data[col].max() - df_data[col].min()) - 1
            df_data.loc[:, f'norm_{col}'] = (1 + df_data.loc[:, f'norm_{col}']) / 2  # (-1,1] -> (0,1]
            df_data.loc[:, f'lognorm_{col}'] = safe_log10(df_data[f'norm_{col}'])

        return df_data.drop(columns=drop_cols), label_enc_dict

    """Prepare args"""
    cat_cols = config.preprocess_params['cat_cols']
    drop_cols = config.preprocess_params['drop_cols']
    date_cols = config.preprocess_params['date_cols']
    numerical_transform_cols = config.preprocess_params['numerical_transform_cols']
    label_col = config.preprocess_params['label_col']

    preprocess_kwargs = dict(
        drop_cols=drop_cols,
        cat_to_num_cols=cat_cols,
        target_col=label_col,
        date_col=date_cols,
        numerical_to_transform_cols=numerical_transform_cols,
    )
    ### ----------------------------------------------------###
    if change_labels:  # if changing labels with loaded data - This part changes the label
        add_label_col(files_df_dict, config)

    data_df = files_df_dict[config.sport_or_casino_task]
    print("Num of N/A rows:", data_df.last_transaction.isna().sum())
    print("Sample Rows: \n", data_df.sample(5))
    print(data_df[data_df['last_transaction'].isna()])

    # if input("Drop Last Transaction data N/A Rows?") in ['Y','y','yes','Yes','YES']:
    data_df.dropna(subset=['last_transaction'], inplace=True)
    print(f"Dropped N/A Rows. Total Rows Now: {data_df.shape[0]:,.0f}")

    """Remove bets after label has reached"""
    if not inference:  # Training Flow
        print('Training and validation Data Labels Distribution')

        if not config.learn_task_vip:
            data_df = data_df[data_df[
                                  'num_deposits'] <= config.active_users_num_deposits]  # Get only data without the label implicitly in the data

        if config.learn_task_vip:
            data_df = data_df[data_df[
                                  'sum_deposit'] <= config.active_users_sum_deposits_vip]  # Get only data without the label implicitly in the data

        df = files_df_dict[config.transactions_df_name].groupby(config.trans_player_id_column, as_index=False)
        if config.learn_task_vip:  # VIP Task
            # sum for each user
            df = df.agg({config.trans_amount_col: 'sum'}).sort_values(config.trans_amount_col, ascending=True)
        else:
            # count for each user
            df = df.agg({config.trans_id_column: 'count'}).sort_values(config.trans_id_column, ascending=True)

        print('All Users')
        print(df.describe())
        clean_filtered_users = df.member_id.sample(int(np.round(df.member_id.nunique() * config.user_ratio_in_test)),
                                                   random_state=42)

        """Sample Test player_id """
        # TODO: Check if needed - duplicate test sample
        test_id = data_df.player_id.sample(int(np.round(data_df.player_id.nunique() * 0.03)),
                                           random_state=1)  # Default - ratio:0.03 random_state:1
        ids_for_test = np.concatenate([clean_filtered_users.values, test_id.values])

        data_df_test = data_df[data_df[config.bets_player_id_column].isin(ids_for_test)]
        data_df = data_df[~data_df[config.bets_player_id_column].isin(ids_for_test)]

    # Preprocess train-val
    preprocess_kwargs['df'] = data_df
    # X,y,label_enc_dict = preprocess_cols(**preprocess_kwargs)
    X, label_enc_dict = preprocess_cols(**preprocess_kwargs)

    if not inference:
        # Preprocess test
        print('Clean Data Labels Distribution')
        preprocess_kwargs['df'] = data_df_test
        # X_clean_test,y_clean_test, label_enc_dict_vip = preprocess_cols(**preprocess_kwargs)
        X_clean_test, label_enc_dict_vip = preprocess_cols(**preprocess_kwargs)
        files_df_dict['df_for_model_test'] = X_clean_test

    files_df_dict['df_for_model'] = X

    return files_df_dict


def main(config_path, inference_flow=False, path_to_file=None,
         label_change_flag=False, test=False, test_frac=0.01, start_block=None):
    config = utils.load_config_from_path(config_path)
    parser = utils.parse_args_from_dict(config)

    if path_to_file:
        files_df_dict = utils.read_from_pickle(path_to_file,
                                               data_type=parser.sport_or_casino_task)  # TODO: Replace with read from DB

        # Test on 1% of Data if test and not sampled at *block_1 - generate_complete_df*
        if start_block < 2 and test:  # If test sampling happens here or happened before
            files_df_dict[parser.sport_or_casino_task] = files_df_dict[parser.sport_or_casino_task].sample(
                frac=test_frac, random_state=42)

        files_df_dict = create_dataset_from_table(files_df_dict, parser,
                                                  inference=inference_flow, change_labels=label_change_flag)

        path = utils.dump_to_pickle(obj=files_df_dict,
                                    filename='block_2_prepare_data_for_model',
                                    data_type=parser.sport_or_casino_task) if not test \
            else utils.dump_to_pickle(obj=files_df_dict,
                                      filename='test_block_2_prepare_data_for_model',
                                      data_type=parser.sport_or_casino_task
                                      )
        return path
    else:
        raise Exception("Error in Path to pickle; Block_2")


if __name__ == '__main__':
    main()
