# import inference
import inference
import prepare_df_for_model
from os.path import join as pjoin
import sys
import datetime

# Local project Imports
import read_data, generate_complete_df
import code_const
import train_model
import test_blocks
from utils import generate_log_file

# TODO: Change random seed to const.
# # (Data needs to be in data folder in working dir if Using STARTING_BLOCK)
STARTING_BLOCK = 0  # Block to start from - * if you want a full flow put 0 *

INFERENCE_FLOW = False  # No training, no labels, no metrics check

FULL_POST_PROCESS = False  # Checking prediction per user and applying voting function for each bet
CHANGE_LABEL = False  # When just changing label and loading data - i.e. not reading csv and generating data

SEARCH_PARAMS = False  # GridsearchCV/RandomCV execution (for train_model block)

TEST = True  # Sampling TEST_FRAC num of bets  - test purpose
TEST_FRAC = 0.1

#########################################################################################
############
# Config Files
config_path = pjoin(code_const.DATA_FOLDER_PATH, code_const.DATA_DIR, 'config_files')
config_path_dict = test_blocks.generate_config_files(config_folder_path=config_path,
                                                     data_type=code_const.SPORT_OR_CASINO_TASK,  # Sport or Casino Data
                                                     learn_task_vip=code_const.TaskVIP,  # VIP or STD Task
                                                     test=TEST)
# Logs
# Print before redirecting prints to log
print_string = f"Current Run: \nData: {code_const.SPORT_OR_CASINO_TASK} \nVIP:{code_const.TaskVIP} "
print_string += f"\nActive Label is :{int(code_const.ActiveUser_LABEL_is_1)}"
if not INFERENCE_FLOW:
    if code_const.TaskVIP:
        print_string += f"\nVIP SUM: {code_const.VIP_SUM_DEPOSITS}"
    print_string += f"\nStarting Block:{STARTING_BLOCK}"
    if TEST:
        print_string += f"\nFor Test/Development - Sampling {TEST_FRAC*100}% of Data."
    if SEARCH_PARAMS:
        print_string += f"\nSearch Parameters: {code_const.SEARCH_PARAMS_TYPE}"
    if INFERENCE_FLOW:
        print_string += f"\nSearch Parameters: {code_const.SEARCH_PARAMS_TYPE}"
else: # INFERENCE FLOW
    print_string += f"\nInference Flow - No Metrics, No Label For Test"

print(print_string)
log_path = pjoin(code_const.DATA_FOLDER_PATH, code_const.DATA_DIR, 'logs')
log = generate_log_file(log_path,
                        data_type=code_const.SPORT_OR_CASINO_TASK,  # Sport or Casino Data
                        learn_task_vip=code_const.TaskVIP,  # VIP or STD Task
                        sum_deposit_vip=code_const.VIP_SUM_DEPOSITS,
                        test=TEST,
                        search=SEARCH_PARAMS,
                        )
sys.stdout = log  # LOGS TO LOG FILE INSTEAD OF PRINTING to STD_OUT

# TODO: For tests - Remove after
data_path_dict = {}
data_path_dict = test_blocks.generate_data_paths(STARTING_BLOCK, code_const.SPORT_OR_CASINO_TASK, TEST)

#############################################################

if __name__ == '__main__':
    start = datetime.datetime.now()
    print(f"Start Time: {start}")
    # BLOCK 0
    path_to_pickle_0 = data_path_dict.get('block_0', None)
    if path_to_pickle_0 is None:
        path_to_pickle_0 = read_data.main(config_path=config_path_dict['config_0'], test=TEST)
    # BLOCK 1
    path_to_pickle_1 = data_path_dict.get('block_1', None)
    if path_to_pickle_1 is None:
        path_to_pickle_1 = generate_complete_df.main(config_path=config_path_dict['config_1'],
                                                     path_to_file=path_to_pickle_0,
                                                     test=TEST, test_frac=TEST_FRAC, inference=INFERENCE_FLOW)
    # BLOCK 2
    path_to_pickle_2 = data_path_dict.get('block_2', None)
    if path_to_pickle_2 is None:
        path_to_pickle_2 = prepare_df_for_model.main(config_path=config_path_dict['config_2'],
                                                     path_to_file=path_to_pickle_1,
                                                     label_change_flag=CHANGE_LABEL, inference_flow=INFERENCE_FLOW,
                                                     test=TEST, test_frac=TEST_FRAC, start_block=STARTING_BLOCK)
    # BLOCK 3
    path_to_pickle_3 = data_path_dict.get('block_3', None)
    if path_to_pickle_3 is None:
        path_to_pickle_3 = train_model.main(config_path=config_path_dict['config_3'], path_to_file=path_to_pickle_2,
                                            test=TEST, search_params=SEARCH_PARAMS, inference=INFERENCE_FLOW)

    # BLOCK 4
    path_to_pickle_4 = inference.main(config_path=config_path_dict['config_4'], path_to_file=path_to_pickle_3,
                                      inference_flow=INFERENCE_FLOW, full_post_process=FULL_POST_PROCESS,
                                      search_params=SEARCH_PARAMS, test=TEST)
    print(f"End Time: {datetime.datetime.now()}, \nTotal Time: {datetime.datetime.now() - start}")
