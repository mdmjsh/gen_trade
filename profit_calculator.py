import logging
import ast
from types import NoneType
from typing import Dict, List, Tuple, Type
from helpers import get_latest_path
import pandas as pd
from run_strategy import win, loss
import argparse
from env import HIT, STOPPED, NA

AMOUNT = 1000


def main(path):
    df = pd.read_csv(path)
    logger = logging.getLogger(__name__)
    res = dict()
    account = 0
    for ix in range(len(df)):
        row = df.iloc[ix]
        logger.info(f"Evaluating {row.id}")
        results = ast.literal_eval(row.result).values()
        for x in results:
            if x == HIT:
                account += win(AMOUNT)
                logger.info(f"target HIT setting account total to {account}")
            elif x == STOPPED:
                account -= loss(AMOUNT)
                logger.info(f"target MISSED setting account total to {account}")
            res[row.id] = account
    return res


def group_consecutive_results(
    path: str | None = None, df: pd.DataFrame | None = None
) -> Dict:
    """groups consecutive results to calculate the cumulative profit and loss.

    NB this would be useful for working out the compounded P&L, but not in use.
    """

    df = df or pd.read_csv(path)
    output = dict()

    for ix in range(len(df)):
        wins = []
        losses = []
        holds = []
        row = df.iloc[ix]
        results = list(ast.literal_eval(row.result).values())
        streak = 1
        for ix, res in enumerate(results):
            win = res == HIT
            loss = res == STOPPED
            hold = res == NA
            switched = False
            try:
                if res == results[ix + 1]:
                    streak += 1
                else:
                    switched = True
                    wins, losses, holds = increment_counters(
                        streak, win, loss, hold, wins, losses, holds
                    )
                    streak = 1
            except IndexError:
                if not switched:
                    wins, losses, holds = increment_counters(
                        streak, win, loss, hold, wins, losses, holds
                    )
                profit = get_profit(wins, losses)
                output[row.id] = dict(
                    wins=wins, losses=losses, holds=holds, profit=profit
                )

    return output


def simple_profit_calculator(
    path: str | None = None, df: pd.DataFrame | None = None
) -> Dict:
    try:
        if len(df):
            df = df
    except (TypeError, ValueError):
        if path:
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=['id', 'results'])
    output = dict()

    for ix in range(len(df)):
        wins = 0
        losses = 0
        holds = 0
        row = df.iloc[ix]
        try:
            results = list(ast.literal_eval(row.result).values())
        except ValueError:
            results = list(row.result.values())
        for res in results:
            if res == HIT:
                wins += 1
            elif res == STOPPED:
                losses += 1
            elif res == NA:
                holds += 1
        output[row.id] = sum(win(AMOUNT) for x in range(wins)) - sum(
            loss(AMOUNT) for x in range(losses)
        )
    return output


def increment_counters(
    streak: int,
    win: bool,
    loss: bool,
    hold: bool,
    wins: List,
    losses: List,
    holds: List,
) -> Tuple:
    if win:
        wins.append(streak)
        losses.append(0)
        holds.append(0)
    elif loss:
        wins.append(0)
        losses.append(streak)
        holds.append(0)
    elif hold:
        wins.append(0)
        losses.append(0)
        holds.append(streak)
    return wins, losses, holds


def get_profit(wins: List, losses: List) -> int:
    zipper = zip(wins, losses)
    profit = 0
    for n_wins, n_losses in zipper:
        if n_wins > n_losses:
            profit += sum(win(AMOUNT) for x in range(n_wins))
        elif n_losses > n_wins:
            profit -= sum(loss(AMOUNT) for x in range(n_losses))
        else:
            profit += 0
    return profit


if __name__ == "__main__":
    path = get_latest_path(".", "csv")
    # logging.basicConfig(
    #     filename=f"{path.split('+')[0]}_trade_log",
    #     filemode="w",
    #     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    #     datefmt="%H:%M:%S",
    #     level=logging.INFO,
    # )
    results = simple_profit_calculator(path)
    import ipdb
    ipdb.set_trace()
