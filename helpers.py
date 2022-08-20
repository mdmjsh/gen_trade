import os
import boto3
import argparse
import pandas as pd
from io import StringIO

from df_adapter import DFAdapter

S3_RESOURCE = boto3.resource("s3", region_name="eu-west-1")
MIN_INDICATORS = int(os.getenv('MIN_INDICATORS', 2))
MAX_SAME_CLASS_INDICATORS = int(os.getenv('MAX_SAME_CLASS_INDICATORS', 2))
MAX_STRATEGY_INDICATORS = int(os.getenv('MAX_STRATEGY_INDICATORS', 4))
POPULATION_SIZE = int(os.getenv('POPULATION_SIZE', 10))
CONJUNCTIONS = ["and", "or", "and not", "or not"]


def make_pandas_df(df: DFAdapter) -> pd.DataFrame:
    # convert dask dd back to pandas
    try:
        return df.compute()
    except AttributeError:
        # df already is a pandas df
        return df


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
