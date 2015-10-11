"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os
import csv

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data


def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """

    # read csv file
    reader = csv.reader(open(orders_file, 'rU'), delimiter=',')

    # eliminate duplicate symbols
    symbols = []
    symbol_dict = {}
    i = 0
    for row in reader:
        if i != 0:
            if (symbol_dict.get(row[1], -1)) == -1:
                symbol_dict[row[1]] = row[1]
                symbols.append(row[1])

        i += 1

    # create dataframes
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices_df = prices_all[symbols]  # only portfolio symbols
    prices_df['Cash'] = 1.0

    count_df = pd.DataFrame(index=prices_df.index, columns=symbols)
    count_df = count_df.fillna(0)

    cash_df = pd.DataFrame(index=prices_df.index, columns=['Cash_Value'])
    cash_df = cash_df.fillna(start_val)

    # populate dataframes
    reader = csv.reader(open(orders_file, 'rU'), delimiter=',')
    i = 0
    for row in reader:
        if i != 0:
            if row[0] in count_df.index:
                if row[2] == 'SELL':
                    count_df.ix[row[0], row[1]] += -float(row[3])
                    cash_df.ix[row[0]] += -float(row[3]) * prices_df.ix[row[0], row[1]]
                if row[2] == 'BUY':
                    count_df.ix[row[0], row[1]] += float(row[3])
                    cash_df.ix[row[0]] += float(row[3]) * prices_df.ix[row[0], row[1]]

        i += 1

    value = start_val
    for date_index, row in count_df.iterrows():
        for i in range(0, len(row), 1):
            if date_index in count_df.index and date_index in prices_df.index:
                value += -(prices_df.ix[date_index, symbols[i]] * count_df.ix[date_index, symbols[i]])
        cash_df.ix[date_index] = value

    count_df['Cash'] = cash_df

    for i in range(0, len(count_df.columns)-1, 1):
        count_df[symbols[i]] = count_df[symbols[i]].cumsum()

    count_df = prices_df * count_df
    count_df['Sum'] = count_df.sum(axis=1)

    # rearrange columns
    columns = count_df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    count_df = count_df[columns]

    # print count_df['Sum']
    return count_df


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-05'
    end_date = '2011-01-20'
    orders_file = os.path.join("orders", "orders-short.csv")
    start_val = 1000000
    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    #Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    #Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    # df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    # plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
