"""# (0) Read Data"""

import os
from os.path import join as pjoin
from zipfile import ZipFile
import argparse
import shutil
import pandas as pd

import utils


# TODO: change config. names
# TODO: add comments to const.
# TODO: change kwargs to JSON File


def read_data(config):
    """
    Reads split .csv files, concatenate them (according to config.files_to_concat)
    :param config: configuration ArgParser
    :return: files_df_dict - dictionary of DFs concatenated
    example:
    files_df_dict = {'CustomersExport_001':         ï»¿CustomerID     CountryName Currency  ... FD_Date   Age gender
    0               51925         Vietnam      USD  ...     NaN   NaN   male
    1               51926  Czech Republic      EUR  ...     NaN   NaN   male
    2                1900  United Kingdom      EUR  ...     NaN  54.0   male
    3               38763          Brazil      EUR  ...     NaN  52.0   male
    4               38764          Latvia      EUR  ...     NaN  46.0   male
    ...               ...             ...      ...  ...     ...   ...    ...
    195054        3096206          Brazil      BRL  ...     NaN  27.0   male
    195055        3096218         Germany      EUR  ...     NaN  52.0   male
    195056        3096273         Germany      EUR  ...     NaN   NaN   male
    195057        3096286         Germany      EUR  ...     NaN  42.0   male
    195058        3096291         Germany      EUR  ...     NaN  28.0   male
    [195059 rows x 10 columns],
     'TransactionsExport':         transaction_id  member_id  ...           created_at         processed_at
    0              7399643    2545390  ...  2022-12-19 04:53:44  2022-12-19 04:54:17
    1              7399605    2545390  ...  2022-12-19 04:41:20  2022-12-19 04:41:54
    2              7399536    2545390  ...  2022-12-19 04:16:58  2022-12-19 04:17:31
    3              7399497    2545390  ...  2022-12-19 04:00:29  2022-12-19 04:01:04
    4              7399415    2545390  ...  2022-12-19 03:30:09  2022-12-19 03:30:41
    ...                ...        ...  ...                  ...                  ...
    218042         8403344    2954024  ...  2023-02-18 16:07:34  2023-02-18 16:08:00
    218043         8250399    1078409  ...  2023-02-09 13:09:13  2023-02-09 13:14:46
    218044         8424945     711770  ...  2023-02-20 01:58:09  2023-02-20 01:58:53
    218045         8328465    2841042  ...  2023-02-14 05:26:58  2023-02-14 08:30:32
    218046         8403306     547681  ...  2023-02-18 16:05:23  2023-02-18 16:06:03
    [1218047 rows x 9 columns],
     'casinoExport':           round_id          round_timestamp  ...  real_bet_amount real_win_amount
    0       1293990690  2022-12-28 06:49:13.000  ...             0.40           -0.40
    1       1293990717  2022-12-28 06:49:14.000  ...             0.40            0.92
    2       1293990740  2022-12-28 06:49:15.000  ...             0.10           -0.10
    3       1293990774  2022-12-28 06:49:17.000  ...             0.20           -0.20
    4       1293990801  2022-12-28 06:49:18.000  ...             2.10           -2.10
    ...            ...                      ...  ...              ...             ...
    499995  1284561032  2022-12-25 12:25:41.000  ...             0.40           -0.40
    499996  1284561049  2022-12-25 12:25:41.000  ...             0.03           -0.03
    499997  1284561074  2022-12-25 12:25:42.000  ...             0.40           -0.40
    499998  1284561109  2022-12-25 12:25:43.000  ...             0.40           -0.16
    499999  1284561125  2022-12-25 12:25:43.000  ...             0.15            0.30
    [26500000 rows x 12 columns],
     'sports':            betid                placed_at  ... Stake_EUR BetWinLoss_EUR
    0       40757424  2022-10-29 13:24:10.000  ...      50.0         -50.00
    1       40694422  2022-10-28 21:51:04.000  ...     100.0         248.00
    2       40692138  2022-10-28 21:20:30.000  ...      31.0         -31.00
    3       40689579  2022-10-28 20:52:30.000  ...     300.0        -300.00
    4       40689459  2022-10-28 20:50:50.000  ...      50.0         -50.00
    ...          ...                      ...  ...       ...            ...
    499995  40691468  2022-10-28 21:13:14.000  ...       6.0           1.50
    499996  40700283  2022-10-28 23:57:23.000  ...      18.0           9.85
    499997  40697613  2022-10-28 22:51:20.000  ...      55.0          78.74
    499998  40693157  2022-10-28 21:33:19.000  ...      22.0         -22.00
    499999  40689484  2022-10-28 20:50:54.000  ...      10.0         -10.00
    [1419555 rows x 14 columns]}
    """
    files_df_dict = dict()
    for file_name in config.data_source:

        if config.unzip_flag:
            extracted_dir = pjoin(config.data_dir, 'extracted')
            zip_filename = file_name + '.csv' + '.zip'  # add zip extension
            with ZipFile(pjoin(config.data_folder_path, config.data_dir, zip_filename), 'r') as zip_f:
                zip_f.extractall(path=extracted_dir)

            data_path = pjoin(extracted_dir, file_name)
            data_df = pd.read_csv(data_path, encoding=config.csv_encoding, on_bad_lines='skip', index_col=False)
            if config.remove_extracted:  # remove zip extracted folder
                shutil.rmtree(extracted_dir)

        ## concat
        elif file_name in config.files_to_concat:
            list_to_merge = []
            for file_n in os.listdir(pjoin(config.data_folder_path, config.data_dir)):  # list files in folder
                if file_n.startswith(file_name) and file_n.endswith('.csv'):  # match file to target filename
                    list_to_merge.append(file_n)
            if len(list_to_merge) > 0:
                path = pjoin(config.data_folder_path, config.data_dir, list_to_merge[0])
                data_df = pd.read_csv(path, encoding=config.csv_encoding, on_bad_lines='skip', index_col=False)
                list_to_merge.pop(0)
                for file_n in list_to_merge:
                    path = pjoin(config.data_folder_path, config.data_dir, file_n)
                    second_df = pd.read_csv(path, encoding=config.csv_encoding, on_bad_lines='skip', index_col=False)
                    data_df = pd.concat([data_df, second_df])  # concat second file to first

                    # Distinct between main_df to others
                if file_name == config.main_df_to_merge:
                    files_df_dict['sports'] = data_df
                else:
                    files_df_dict[file_name] = data_df

        else:
            data_df = pd.read_csv(pjoin(config.data_folder_path, config.data_dir, file_name + '.csv'),
                                  encoding=config.csv_encoding,
                                  index_col=False)

            # Distinct between main_df to others
        if file_name == config.main_df_to_merge:
            files_df_dict['sports'] = data_df
        else:
            files_df_dict[file_name] = data_df

    # TODO: comment on return
    return files_df_dict

    # TODO: add description for each function in main


def main(config_path, test=False):
    config = utils.load_config_from_path(config_path)
    parser = utils.parse_args_from_dict(config)

    files_df_dict = read_data(parser)
    utils.dump_df_dict(files_df_dict)
    # TODO: replace with save to DB
    path = utils.dump_to_pickle(obj=files_df_dict,
                                filename='block_0_read_data',
                                data_type=parser.sport_or_casino_task) if not test \
        else utils.dump_to_pickle(obj=files_df_dict,
                                  filename='test_block_0_read_data',
                                  data_type=parser.sport_or_casino_task)
    return path

# if __name__ == '__main__':
#     main()
