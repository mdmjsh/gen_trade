from importlib.resources import path
import os
import boto3
import argparse
import pandas as pd
from io import StringIO

from df_adapter import DFAdapter
from env import (
    S3_RESOURCE,
    POPULATION_SIZE,
    MAX_STRATEGY_INDICATORS,
    MAX_SAME_CLASS_INDICATORS,
)


def make_pandas_df(df: DFAdapter) -> pd.DataFrame:
    # convert dask dd back to pandas
    try:
        return df.compute()
    except AttributeError:
        # df already is a pandas df
        return df


def get_latest_path(path: str, suffix: str | None = None) -> path:
    paths = []
    for entry in os.scandir(path):
        if entry.is_dir():
            continue
        if suffix and entry.name.endswith(suffix):
            paths.append(entry)
    return max(paths, key=os.path.getctime).path


def write_df_to_s3(df: pd.DataFrame, bucket: str, name: str):
    """write csv to s3 bucket. Overwrites file if already exists."""
    buffer = StringIO()
    df.to_csv(buffer)
    S3_RESOURCE.Object(bucket, name).put(Body=buffer.getvalue())


def base_arg_parser(help_message: str = None):
    parser = argparse.ArgumentParser(help_message)
    parser.add_argument(
        "--population_size",
        type=int,
        help="number of strategies to generate",
        required=False,
    )
    parser.add_argument(
        "--max_indicators",
        type=int,
        help="max number of indicators in a strategy",
        required=False,
    )
    parser.add_argument(
        "--max_same_class",
        type=int,
        help="max number of same class indicators in a strategy",
        required=False,
    )

    parser.set_defaults(
        population_size=POPULATION_SIZE,
        max_indicators=MAX_STRATEGY_INDICATORS,
        max_same_class=MAX_SAME_CLASS_INDICATORS,
    )
    return parser
