import sys
import logging
import os
import argparse
import pandas as pd
from binance.client import Client
from datetime import datetime


def write_data(klines, csv_path):
    _klines = [k[:-3] for k in klines] # Drop some of the columns we're not interested in
    df = pd.DataFrame(_klines)
    df.columns = ['open_ts','open', 'high', 'low', 'close', 'volume','close_ts', 'qav','num_trades']

    df.index = [datetime.fromtimestamp(x/1000.0) for x in df.close_ts]
    df.index = [datetime.fromtimestamp(x/1000.0) for x in df.open_ts]

    for y in [df.open_ts, df.close_ts]:
        for x in y:
            df.index = [datetime.fromtimestamp(x/1000.0)]
    # nb timestamps are unix time, could optionally convert by * 1000, but maybe not necessary?
    df.to_csv(csv_path)

def main(start_date, end_date, symbol_a, symbol_b, csv_path):
    client = Client(os.environ["BINANCE_API"], os.environ["BINANCE_SECRET"])
    klines = client.get_historical_klines(f"{symbol_a}{symbol_b}", client.KLINE_INTERVAL_15MINUTE, start_date, end_date)
    write_data(klines, csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download historical Binance data.")
    parser.add_argument('start_date', type=str, help="Start date - human readable (see Binance API docs) - default `1 Jan, 2018`" )
    parser.add_argument('end_date', type=str, help="- human readable (see Binance API docs) default `1 day ago UTC`")
    parser.add_argument('symbol_a', type=str, help='First ticker in the trade - default BTC')
    parser.add_argument('symbol_b', type=str, help='Secondary ticker in the trade - default USDC')
    parser.add_argument('csv_path', type=str, help='Output path/filename for CSV. If None uses the concat of the symbols.')

    parser.set_defaults(start_date="1 Jan, 2018", end_date="1 day ago UTC", symbol_a="BTC", symbol_b="USDC")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s', stream=sys.stderr, level=logging.ERROR)

    csv_path = args.csv_path or f"{args.symbol_a}{args.symbol_b}.csv"
    main(args.start_date, args.end_date, args.symbol_a, args.symbol_b, csv_path)
