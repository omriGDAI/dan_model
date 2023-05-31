"""# Args

## Source CONST
"""

DATA_FOLDER_PATH = "/Users/omrilapidot/PycharmProjects/dan_model/data"
SOURCE = 'Delasport'
TRANSACTIONS_DF_NAME = 'TransactionsExport'
LoadData = True

# Transactions Table
TRANSACTIONS_DF_NAME = 'TransactionsExport'
DEPOSIT_WITHDRAW_COLUMN_NAME = 'type'
DEPOSIT_NAME_IN_COLUMN = 'deposit'
TRANSACTION_STATUS_COLUMN_NAME = 'status'
TRANS_APPROVED_IN_COLUMN = 'approved'
TRANS_PLAYER_ID_COLUMN = 'member_id'
TRANS_ID_COLUMN = 'transaction_id'
TRANS_AMOUNT_COL = 'amount_EUR'
TRANS_DATE_COLUMN_NAME = 'created_at'

# Learning Task Indicators
RATIO_USER_IN_TEST = 0.2
VAL_PLAYERS_RATIO = 0.3  # POC with 0.3 in sports

LABEL_COLUMN_NAME = 'active_user'
ActiveUser_LABEL_is_1 = True
ACTIVE_NUM_DEPOSITS = 1  # not inclusive

SPORT_OR_CASINO_TASK = 'sports'  # 'sports' , 'casinoExport'
TaskVIP = True
VIP_SUM_DEPOSITS = 1000  # 1000 EUR
SUM_DEPOSIT_COLUMN_NAME = 'sum_deposit'
NUM_DEPOSITS_COLUMN_NAME = 'num_deposits'

# Train_model-block parameters
XGB_BEST_THRESHOLD = 0.5
PLOT_PROB_THRESH = False
PLOT_THRESH_NUM_SAMPLES = 50


PLOT_METRICS_AND_IMPORTANCE = False
SEARCH_FOR_PARAMS_FLAG = False #TODO: Duplicate
SEARCH_PARAMS_TYPE = 'random'  # 'random' or 'grid'
RegTask = False
VOTING_THR = 0.5

# XGB params per task per data
if SPORT_OR_CASINO_TASK == 'sports':  # Sport Data
    if TaskVIP:
        if VIP_SUM_DEPOSITS == 5000:
            XGB_TRAIN_PARAMS = dict(
                n_jobs=-1,
                objective="binary:logistic",  # if not RegTask else "reg:squarederror",
                random_state=42,
                eval_metric=["error@0.5"],
                # ,"error@0.2","auc", "aucpr", "error"],  # "error", "error@0.6","error@0.7","error@0.8","error@0.9",
                verbosity=1,  # 2
                early_stopping_rounds=30,
                n_estimators=5000,  # 800
                colsample_bytree=0.9,
                colsample_bylevel=0.9,
                subsample=0.5,
                # gamma = 0.2,
                max_depth=2,
                min_child_weight=1,
                # learning_rate = 0.2,
                reg_lambda=30,
                reg_alpha=15,
                # max_delta_step = 5,
                # eta = 10e-5,
                # grow_policy='lossguide',
                max_bin=2048,
            )
        elif VIP_SUM_DEPOSITS == 1000:
            XGB_TRAIN_PARAMS = dict(
                n_jobs=-1,
                objective="binary:logistic",
                random_state=42,
                eval_metric=["error@0.2", "auc", "aucpr", "error"],
                verbosity=1,  # 2
                early_stopping_rounds=30,
                n_estimators=5000,  # 800
                colsample_bytree=0.9,
                colsample_bylevel=0.9,
                subsample=0.5,
                max_depth=2,
                min_child_weight=1,
                reg_lambda=30,
                reg_alpha=15,
                max_bin=2048,
            )
        else:
            raise Exception(f"Sports VIP Task - Not trained on {VIP_SUM_DEPOSITS} EUR VIP")
    else:  # STD Task
        XGB_TRAIN_PARAMS = {'n_jobs': -1, 'objective': 'binary:logistic', 'random_state': 42, 'eval_metric': ['aucpr'],
                            'verbosity': 1, 'early_stopping_rounds': 30, 'n_estimators': 200, 'colsample_bytree': 0.9,
                            'colsample_bylevel': 0.75, 'subsample': 0.25, 'gamma': 20, 'max_depth': 7,
                            'reg_lambda': 27000,
                            'max_bin': 2048, 'tree_method': 'hist'}

if SPORT_OR_CASINO_TASK == 'casinoExport':  # Casino Data
    if TaskVIP:
        if VIP_SUM_DEPOSITS == 50000:
            XGB_TRAIN_PARAMS = dict(
                n_jobs=-1,
                objective="binary:logistic",  # if not RegTask else "reg:squarederror",
                random_state=42,
                eval_metric=["error@0.5"],
                # ,"error@0.2","auc", "aucpr", "error"],  # "error", "error@0.6","error@0.7","error@0.8","error@0.9",
                verbosity=1,  # 2
                early_stopping_rounds=30,
                n_estimators=5000,  # 800
                colsample_bytree=0.9,
                colsample_bylevel=0.9,
                subsample=0.5,
                # gamma = 0.2,
                max_depth=2,
                min_child_weight=1,
                # learning_rate = 0.2,
                reg_lambda=30,
                reg_alpha=15,
                # max_delta_step = 5,
                # eta = 10e-5,
                # grow_policy='lossguide',
                max_bin=2048,
            )
        elif VIP_SUM_DEPOSITS == 10000:
            XGB_TRAIN_PARAMS = {'colsample_bylevel': 0.9719063155284207,
                                'colsample_bynode': 0.7333592446918453,
                                'colsample_bytree': 0.7970500417163436,
                                'gamma': 0.2838411191854767,
                                'learning_rate': 0.23746425679765054,
                                'max_depth': 4,
                                'n_estimators': 59,
                                'objective': 'binary:logistic',
                                'random_state': 42,
                                'reg_alpha': 2.3763583253614384,
                                'reg_lambda': 35.257874033131465,
                                'subsample': 0.5947630906348191,
                                'tree_method': 'hist'}
        else:
            raise Exception(f"Casino VIP Task - Not trained on {VIP_SUM_DEPOSITS} EUR VIP")
    else:  # STD Task
        XGB_TRAIN_PARAMS = {'n_jobs': -1, 'objective': 'binary:hinge', 'random_state': 42, 'eval_metric': ['error'],
                            'verbosity': 1, 'early_stopping_rounds': 50, 'n_estimators': 5000,
                            'colsample_bytree': 0.8, 'colsample_bynode': 0.9, 'max_depth': 4, 'learning_rate': 0.05,
                            'max_bin': 2048,
                            'tree_method': 'hist'}

XGB_TRAIN_PARAMS = dict(
    n_jobs=-1,
    objective="binary:logistic",  # if not RegTask else "reg:squarederror",
    random_state=42,
    eval_metric=["auc", "error", "aucpr"],  # "error", "error@0.6","error@0.7","error@0.8","error@0.9",
    verbosity=1,  # 2
    early_stopping_rounds=30,
    n_estimators=5000,  # 800
    colsample_bytree=0.9,
    subsample=0.5,
    max_depth=4,
    tree_method='hist',
)
### RANDOM CV
RANDOM_CV_XGB_PARAMS = {  # if int then uses randint, if float uses uniform, else static var
    'objective': ["binary:logistic"],
    "colsample_bytree": {'start': 0.6, 'interval': 0.4},
    "gamma": {'start': 0.0, 'interval': 25.0},
    "learning_rate": {'start': 0.05, 'interval': 0.4},
    "max_depth": {'start': 3, 'end': 9},
    "n_estimators": {'start': 40, 'end': 80},
    "subsample": {'start': 0.4, 'interval': 0.3},
    "reg_lambda": {'start': 0.0, 'interval': 300.0},
    "reg_alpha": {'start': 0.0, 'interval': 20.0},
    'tree_method': ['hist', 'exact'],
    'colsample_bylevel': {'start': 0.7, 'interval': 0.3},
    'colsample_bynode': {'start': 0.7, 'interval': 0.3},
    'random_state': [42],
}

RANDOM_CV_META_PARAMS = dict(
    random_state=42,
    n_iter=200,
    cv=3,
    verbose=True,
    n_jobs=-1,
    return_train_score=True,
    scoring='f1')
### GRIDSEARCH
GRIDSEARCH_XGB_PARAMS = {
    "colsample_bytree": [0.8],  # [0.7,0.8,0.9], #uniform(0.7, 0.3),
    "gamma": {'start': 0.0, 'end': 10.0, 'steps': 5, 'log_scale': True},
    "max_depth": [2, 3, 4, 5, 6],  # default 6
    "n_estimators": [30],  # default 100
    "subsample": [0.5],  # [0.6,0.7,0.8], #uniform(0.6, 0.25),
    "reg_lambda": {'start': 0.1, 'end': 10.0, 'steps': 5, 'log_scale': True},
    "reg_alpha": {'start': 0.0, 'end': 10.0, 'steps': 5, 'log_scale': True},
    'learning_rate': [0.3, 0.1],
    'tree_method': ['hist'],
}
GRIDSEARCH_META_PARAMS = dict(
    cv=2,
    verbose=True,
    n_jobs=-1,
    return_train_score=True,
    scoring='f1')

"""### Concat Args"""

DATAFRAMES_TO_MERGE_TO = ['sports', 'casinoExport']
SORT_BY_COL = 'ï»¿CustomerID'
MAIN_DF_COL_TO_MAP_BY = 'player_id'
FILENAMES_TO_MERGE_FROM = ['CustomersExport_001']
COLS_TO_MERGE_PER_FILENAME = \
    {'CustomersExport_001': ['CountryName', 'Age', 'Currency', 'account_status', 'FromAffiliate',
                             'FirstProduct_Preference', 'gender'],
     }

"""### Pre-process Args"""

if SPORT_OR_CASINO_TASK == 'sports':  # Sport data

    # feature engineering params
    BET_ID_COLUMN = 'betid'
    BETS_PLAYER_ID_COLUMN = 'player_id'
    BET_DATE_COLUMN = ' placed_at'

    # preprocess params
    preprocess_params = dict(
        cat_cols=['bet_type', ' bet_mode', 'CountryName', 'Currency', 'account_status', 'FromAffiliate'
            , 'FirstProduct_Preference', 'gender'],

        drop_cols=[BET_ID_COLUMN, 'SportName', 'LeagueName', 'account_status', 'gender'],  # + ['diff_1']
        # drop_cols = drop_cols + ['Unnamed: 0'] if 'Unnamed: 0' in data_df.columns else drop_cols
        # drop_cols += ['active_user',PLAYER_ID_COLUMN]

        date_cols=[BET_DATE_COLUMN, 'last_transaction'],

        numerical_transform_cols=['Stake_EUR', 'BetWinLoss_EUR', 'last_deposit_amount', 'sum_deposit'],

        label_col='active_user'
    )
######---------------------------#########################

elif SPORT_OR_CASINO_TASK == 'casinoExport':  # Casino data
    # feature engineering params
    BET_ID_COLUMN = 'round_id'
    BETS_PLAYER_ID_COLUMN = 'player_id'
    BET_DATE_COLUMN = 'round_timestamp'

    # preprocess params
    preprocess_params = dict(
        cat_cols=['CountryName', 'player_currency', 'Currency', 'account_status', 'FromAffiliate'
            , 'FirstProduct_Preference', 'gender'],

        drop_cols=[BET_ID_COLUMN, 'game_name', 'vendor_name', 'account_status'],  # + ['diff_1']
        # drop_cols = drop_cols + ['Unnamed: 0'] if 'Unnamed: 0' in data_df.columns else drop_cols
        # drop_cols += ['active_user',PLAYER_ID_COLUMN]

        date_cols=[BET_DATE_COLUMN, 'last_transaction'],

        numerical_transform_cols=['bonus_bet_amount', 'real_bet_amount', 'last_deposit_amount', 'sum_deposit'],

        label_col='active_user'
    )

"""### Read Data Block"""

cols_to_merge_dict = dict()

if SOURCE == 'Delasport':
    DATA_DIR = 'delasport/data'
    DATA_SOURCE = ["CustomersExport_001", "SportExport", "TransactionsExport", "casinoExport"]
    # ,"Freebet_n_Cashback_Bonuses_Export" ,"DepositBonuses_Export"]
    FILES_TO_CONCAT = ["SportExport", "TransactionsExport", "casinoExport"]
    CSV_ENCODING = 'windows-1254'
    UNIQUE_COUNT_COL = 'origin'
    UNZIP_FLAG = False
    REMOVE_EXTRACTED = True

    MAIN_DF_TO_MERGE = "SportExport"

    # Pre-process CONST
    filename = "TransactionsExport"
    COLS_TO_MERGE = ['sport_id', 'dela_sport_name']
    COLS_TO_MERGE += ['league', 'league_id']
    COLS_TO_MERGE += ['event_home_team', 'event_away_team', 'event_home_team_id', 'event_away_team_id']

    # COLS_TO_SEPERATE_BY = [['event_home_team','event_home_team_id'],['event_away_team_id','event_away_team']]

    SECONDARY_DF_COL_TO_MAP_BY = 'event_id'
    # add to dictionary
    cols_to_merge_dict[filename] = {
        "col_map_by": SECONDARY_DF_COL_TO_MAP_BY,
        "cols_to_merge": COLS_TO_MERGE,
        # "cols_to_seperate":COLS_TO_SEPERATE_BY,
    }
