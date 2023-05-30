import argparse
import json

"""###### Utils"""

from os.path import join as pjoin
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
import pickle
import os
import sys
import xgboost as xgb


def parse_args_from_dict(kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    parser = parser.parse_args()
    return parser


def dump_to_pickle(obj, filename, data_type):
    path = pjoin('Data', data_type)
    if not os.path.exists(path):  # create folders
        os.makedirs(path)
    path = pjoin(path, filename)
    print(f"Dumping file {path} to pickle...")
    pickle_out = open(f"{path}.pickle", "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    return os.path.realpath(pickle_out.name)


def read_from_pickle(path, data_type):
    pickle_in = open(path, "rb")
    obj = pickle.load(pickle_in)
    return obj


def read_model(path):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def load_config_from_path(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


def generate_log_file(log_path, data_type, learn_task_vip, search=False, sum_deposit_vip=None, test=False):
    path = pjoin(log_path, data_type)
    if learn_task_vip:
        path = pjoin(path, f'VIP_{sum_deposit_vip}')
    else:
        path = pjoin(path, f'STD')

    if test:
        path = pjoin(path, 'TEST')

    if not os.path.exists(path):  # create folders
        os.makedirs(path)

    filename = "run_log.log" if not search else "search_run_log.log"
    path = pjoin(path, filename)
    log = open(path, "w")
    return log
