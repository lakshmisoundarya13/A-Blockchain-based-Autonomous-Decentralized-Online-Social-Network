# Install required packages (use pip in terminal, not in script)
# pip install stargazer pingouin statsmodels
# pylint: disable=unused-import,unused-variable,missing-docstring,invalid-name,
# wrong-import-order, ungrouped-imports, too-many-arguments, line-too-long
# Imports
from pathlib import Path
from google.cloud import bigquery
import numpy as np # type: ignore
import pandas as pd
import decimal
from datetime import datetime, date, timedelta, timezone
from dateutil.relativedelta import relativedelta
import math
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import statsmodels.api as sm
import plotly.colors as pc
import re
from stargazer.stargazer import Stargazer
import pingouin
import IPython.core.display
import requests
import json
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from math import sqrt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from google.oauth2 import service_account

# Replace with the path to your downloaded JSON credentials file
key_path = r"C:\Users\laksh\OneDrive\Desktop\Project\prefab-envoy-466116-m1-c8d8082042ac.json"
# Create credentials object and BigQuery client
credentials = service_account.Credentials.from_service_account_file(key_path)
print("Using service account file:", key_path)
client = bigquery.Client(credentials=credentials, project="trans-invention-305714")
PROJECT_ID = "trans-invention-305714"
#client = bigquery.Client(project=PROJECT_ID, location="US")

# Access dataset and print table names
dataset_ref = client.dataset("crypto_ethereum", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)


# Decentralization class
class Decentralization:
    """
    Class to calculate decentralization index for a given ERC-20 token address
    using entropy of transaction volumes.

    Parameters:
    - start (datetime.date): Start date of querying
    - end (datetime.date): End date of querying
    - address (str): ERC-20 token contract address
    """

    def __init__(self, start, end, address):
        self.start = start
        self.end = end
        self.address = address

    def program(self):
        df_program = pd.date_range(start=self.start, end=self.end)
        duration = (self.end - self.start).days + 1  # Added to fix undefined 'duration'
        days = np.size(duration)
        Entropy = pd.DataFrame(np.zeros(days), columns=["val"])
        Entropy["date"] = df_program

        for i in range(0, days):
            start_date = self.start + timedelta(days=i)
            end_date = start_date + timedelta(days=1)

            sql = f"""
                SELECT token_address, from_address, to_address, block_timestamp,
                       CAST(value AS NUMERIC) AS value
                FROM `bigquery-public-data.crypto_ethereum.token_transfers`
                WHERE token_address = '{self.address}'
                AND CAST(value AS NUMERIC) <> 0
                AND block_timestamp >= TIMESTAMP('{start_date} 00:00:00+00')
                AND block_timestamp < TIMESTAMP('{end_date} 00:00:00+00')
            """

            df = client.query(sql).to_dataframe(progress_bar_type="tqdm_notebook")
            Ent = self.processing_tranvol(df)
            Entropy.loc[i, "val"] = Ent

        return Entropy

    def processing_tranvol(self, df):
        df.rename(
            columns={"f0_": "value", "from_address": "from", "to_address": "to"},
            inplace=True,
        )
        df["value"] = df["value"].astype(float)
        df = df.dropna()
        E = self.index_1(df)
        return E

    def index_1(self, df):
        df["pr"] = df["value"] / df["value"].sum()
        q = -df["pr"] * np.log2(df["pr"])
        Entropy_sum = q.sum()
        v = 2**Entropy_sum
        return v
