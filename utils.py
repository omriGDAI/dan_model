import argparse
import json
from sqlalchemy import create_engine
import psycopg2

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


def dump_df_to_db(df, table_name):
    # Define the connection information
    db_endpoint = "demo-db.ctk9oi0waozt.eu-north-1.rds.amazonaws.com"
    db_port = 5432
    db_username = "GDAI_ADMIN"
    db_password = "7fq_{R):.#!3+[D12eP<<lJV+6>y"
    db_name = "postgres"  # Replace with your actual database name

    # Establish a connection to the PostgreSQL database
    engine = create_engine(f"postgresql://{db_username}:{db_password}@{db_endpoint}:{db_port}/{db_name}")

    # Write the DataFrame to the PostgreSQL database table
    table_name = table_name
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    # Close the database connection
    engine.dispose()

    print("Data has been written to the PostgreSQL database successfully!")


def dump_df_dict(df_dict: dict):
    for df_name, df in df_dict.items():
        if isinstance(df, pd.DataFrame):
            dump_df_to_db(df, df_name)


def read_table_from_db(table_name):
    conn = psycopg2.connect(
        host="demo-db.ctk9oi0waozt.eu-north-1.rds.amazonaws.com",
        port=5432,
        database="postgres",
        user="GDAI_ADMIN",
        password="7fq_{R):.#!3+[D12eP<<lJV+6>y"
    )

    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df


def read_dict_from_db():
    db_dict = {}
    tabels_names = ["CustomersExport_001", "sports", "casinoExport", "TransactionsExport"]
    for tabels_name in tabels_names:
        db_dict[tabels_name] = read_table_from_db(tabels_name)
    return db_dict
