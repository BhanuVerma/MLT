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
    # pd.set_option('display.max_rows', len(prices_df))
    # print prices_df
    count_df = pd.DataFrame(index=prices_df.index, columns=symbols)
    count_df = count_df.fillna(0)

    cash_df = pd.DataFrame(index=prices_df.index, columns=['Cash_Value'])
    cash_df = cash_df.fillna(start_val)

    leverage_df = pd.DataFrame(index=prices_df.index, columns=['Leverage'])
    leverage_df = leverage_df.fillna(0)

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

    symbols_sum = []
    for i in range(len(symbols)):
        symbols_sum.append(0)

    for date_index, row in count_df.iterrows():
        longs = 0
        shorts = 0
        for i in range(0, len(row), 1):
            if date_index in count_df.index and date_index in prices_df.index:
                symbols_sum[i] += count_df.ix[date_index, symbols[i]]
                # print date_index, symbols_sum[i]
                value += -(prices_df.ix[date_index, symbols[i]] * count_df.ix[date_index, symbols[i]])
                if symbols_sum[i] > 0:
                    longs += (prices_df.ix[date_index, symbols[i]] * symbols_sum[i])
                if symbols_sum[i] < 0:
                    shorts += abs((prices_df.ix[date_index, symbols[i]] * symbols_sum[i]))
        leverage = (longs + shorts)/(longs - shorts + value)
        leverage_df.ix[date_index] = leverage
        if leverage > 2.0:
            longs = 0
            shorts = 0
            # print "Raise Alert"
            # print date_index, leverage, temp_value
            for i in range(0, len(symbols_sum), 1):
                symbols_sum[i] -= count_df.ix[date_index, symbols[i]]

            for i in range(0, len(symbols_sum), 1):
                if symbols_sum[i] > 0:
                    longs += (prices_df.ix[date_index, symbols[i]] * symbols_sum[i])
                if symbols_sum[i] < 0:
                    shorts += abs((prices_df.ix[date_index, symbols[i]] * symbols_sum[i]))
            previous_leverage = (longs + shorts)/(longs - shorts + value)

            # print leverage, previous_leverage
            if leverage > previous_leverage > 2.0:
                leverage_df.ix[date_index] = previous_leverage
                cash_df.ix[date_index] = value
                temp_value = value
            else:
                count_df.ix[date_index] = 0
                cash_df.ix[date_index] = temp_value
                value = temp_value
        else:
            cash_df.ix[date_index] = value
            temp_value = value

    count_df['Cash'] = cash_df
    count_df['Leverage'] = leverage_df

    # print
    # pd.set_option('display.max_rows', len(count_df))
    # print count_df
    # print
    # find cumulative sum
    for i in range(0, len(symbols), 1):
        count_df[symbols[i]] = count_df[symbols[i]].cumsum()
    # print count_df
    # print
    # dot product of matrices
    count_df = prices_df * count_df
    count_df['Sum'] = count_df.sum(axis=1)
    # print count_df

    # rearrange columns
    columns = count_df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    count_df = count_df[columns]

    # pd.set_option('display.max_rows', len(count_df['Sum']))
    # print count_df['Sum']
    return count_df


# THIS FUNCTION WOULD NOT PASS THE LEVERAGE TEST CASES

# def compute_portvals(start_date, end_date, orders_file, start_val):
#     """Compute daily portfolio value given a sequence of orders in a CSV file.
#
#     Parameters
#     ----------
#         start_date: first date to track
#         end_date: last date to track
#         orders_file: CSV file to read orders from
#         start_val: total starting cash available
#
#     Returns
#     -------
#         portvals: portfolio value for each trading day from start_date to end_date (inclusive)
#     """
#
#     # read csv file
#     reader = csv.reader(open(orders_file, 'rU'), delimiter=',')
#
#     # eliminate duplicate symbols
#     symbols = []
#     symbol_dict = {}
#     i = 0
#     for row in reader:
#         if i != 0:
#             if (symbol_dict.get(row[1], -1)) == -1:
#                 symbol_dict[row[1]] = row[1]
#                 symbols.append(row[1])
#
#         i += 1
#
#     # create dataframes
#     dates = pd.date_range(start_date, end_date)
#     prices_all = get_data(symbols, dates)  # automatically adds SPY
#     prices_df = prices_all[symbols]  # only portfolio symbols
#     prices_df['Cash'] = 1.0
#
#     count_df = pd.DataFrame(index=prices_df.index, columns=symbols)
#     count_df = count_df.fillna(0)
#
#     cash_df = pd.DataFrame(index=prices_df.index, columns=['Cash_Value'])
#     cash_df = cash_df.fillna(start_val)
#
#     # populate dataframes
#     reader = csv.reader(open(orders_file, 'rU'), delimiter=',')
#     i = 0
#     for row in reader:
#         if i != 0:
#             if row[0] in count_df.index:
#                 if row[2] == 'SELL':
#                     count_df.ix[row[0], row[1]] += -float(row[3])
#                     cash_df.ix[row[0]] += -float(row[3]) * prices_df.ix[row[0], row[1]]
#                 if row[2] == 'BUY':
#                     count_df.ix[row[0], row[1]] += float(row[3])
#                     cash_df.ix[row[0]] += float(row[3]) * prices_df.ix[row[0], row[1]]
#
#         i += 1
#
#     value = start_val
#     for date_index, row in count_df.iterrows():
#         for i in range(0, len(row), 1):
#             if date_index in count_df.index and date_index in prices_df.index:
#                 value += -(prices_df.ix[date_index, symbols[i]] * count_df.ix[date_index, symbols[i]])
#         cash_df.ix[date_index] = value
#
#     count_df['Cash'] = cash_df
#
#     print prices_df
#     print
#     print count_df
#     print
#     # find cumulative sum
#     for i in range(0, len(count_df.columns)-1, 1):
#         count_df[symbols[i]] = count_df[symbols[i]].cumsum()
#     print count_df
#     print
#     # dot product of matrices
#     count_df = prices_df * count_df
#     count_df['Sum'] = count_df.sum(axis=1)
#     print count_df
#
#     # rearrange columns
#     columns = count_df.columns.tolist()
#     columns = columns[-1:] + columns[:-1]
#     count_df = count_df[columns]
#
#     return count_df


# def test_run():
#     ordersFile = os.path.join('orders', 'leverageTest1.csv')
#     leo_tester(startDate='2011-01-03', endDate='2011-12-14', ordersFile=ordersFile)
#     ordersFile = os.path.join('orders', 'leverageTest2.csv')
#     leo_tester(startDate='2011-01-03', endDate='2011-12-14', ordersFile=ordersFile)
#     ordersFile = os.path.join('orders', 'leverageTest3.csv')
#     leo_tester(startDate='2011-01-03', endDate='2011-12-14', ordersFile=ordersFile)
#     ordersFile = os.path.join('orders', 'leverageTest4.csv')
#     leo_tester(startDate='2011-01-03', endDate='2011-12-14', ordersFile=ordersFile)


# def leo_tester(startDate, endDate, ordersFile, resultsFile=None):
#     ''' Enhanced testing funtion. Mostly works the same as Tucker's... but it's way better
#
#     Parameters
#     ----------
#         startDate: (str) first date to track
#         endDate: (str) last date to track
#         ordersFile: (str) CSV file to read orders from, ACTUALLY THIS REQUIRES A PATH
#         resultsFile: (str) filepath for saving down answers, defaults to None
#
#     Returns
#     -------
#         None
#     '''
#     start_val = 1000000
#     # Process orders
#     portvals = compute_portvals(startDate, endDate, ordersFile, start_val)
#     if isinstance(portvals, pd.DataFrame):
#         portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
#     # Get portfolio stats
#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
#
#     if resultsFile is not None:
#         portvals.to_csv(resultsFile)
#
#     # check answers
#     if ordersFile == os.path.join('orders', 'leverageTest1.csv'):
#         ansFile = os.path.join('orders', 'leverageTest1_ans.csv')
#         testVsAnswer(portvals, ansFile, ordersFile)
#     elif ordersFile == os.path.join('orders', 'leverageTest2.csv'):
#         ansFile = os.path.join('orders', 'leverageTest2_ans.csv')
#         testVsAnswer(portvals, ansFile, ordersFile)
#     elif ordersFile == os.path.join('orders', 'leverageTest3.csv'):
#         ansFile = os.path.join('orders', 'leverageTest3_ans.csv')
#         testVsAnswer(portvals, ansFile, ordersFile)
#     elif ordersFile == os.path.join('orders', 'leverageTest4.csv'):
#         ansFile = os.path.join('orders', 'leverageTest4_ans.csv')
#         testVsAnswer(portvals, ansFile, ordersFile)


# def testVsAnswer(portvals, ansFile, ordersFile):
#     ''' Testing file for individual answers, tries to tell you what you got wrong
#
#     Parameters
#     ----------
#         portvals: (pd.Series) portfolio values series
#         ansFile: (str) CSV file path to read answers from
#         ordersFile: (str) CSV file path to read orders from,
#
#     Returns
#     -------
#         None
#     '''
#     print '*****************************************'
#     print '   Testing %s' % ordersFile
#     ansSeries = pd.Series.from_csv(ansFile)
#     shapeEqual = portvals.shape == ansSeries.shape
#     if not shapeEqual:
#         print '******* ERROR in test: %s ********' % ordersFile
#         print '    user answer: %s' % portvals.shape
#         print '    target answer: %s' % ansSeries.shape
#     else:
#         ax = ansSeries.plot(label='answer')
#         portvals.plot(ax=ax, linestyle='--', label='user', color='r')
#         ax.set_title('Test for %s' % ordersFile)
#         ax.legend()
#         # from matplotlib import pyplot as plt
#         # plt.show()
#         try:
#             np.testing.assert_array_almost_equal(portvals.values, ansSeries.values, decimal=4)
#             print '****** SUCCESS!  CONGRATULATIONS! *******'
#         except:
#             print '******* ERROR in test: %s ********' % ordersFile
#             print '    see graphs to debug'
#     return None


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-10'
    end_date = '2011-12-20'
    orders_file = os.path.join("orders", "orders.csv")
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
