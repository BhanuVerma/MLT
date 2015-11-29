# __author__ = 'Bhanu Verma'

import os
import math
import pandas as pd
import matplotlib.pyplot as plot
import csv
import numpy as np


class LinRegLearner(object):
    def __init__(self):
        pass  # move along, these aren't the drones you're looking for

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:, 0:dataX.shape[1]] = dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]


class KNNLearner(object):
    def __init__(self, k):
        if k <= 0:
            k = 3
        self.n_n = k
        self.x_data = None
        self.y_data = None

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # load data into class variables
        self.x_data = dataX
        self.y_data = dataY

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        row_count, column_count = points.shape
        result_y = np.zeros(row_count)
        for count in range(row_count):
            a = points[count]
            euc_dist = np.sqrt(((a - self.x_data) ** 2).sum(axis=1))
            sorted_indices = euc_dist.argsort()
            n_neighbours = sorted_indices[:self.n_n]
            nearest_y = [self.y_data[index] for index in n_neighbours]
            result_y[count] = np.mean(nearest_y)

        return result_y


class BagLearner(object):
    def __init__(self, learner, kwargs, bags=20, boost=False):
        if bags <= 0:
            bags = 20
        if 'k' in kwargs:
            if kwargs['k'] <= 0:
                kwargs['k'] = 3
        else:
            kwargs['k'] = 3
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY):
        row_count, column_count = dataX.shape
        count = 0
        for l in self.learners:
            sampled_x = np.zeros(dataX.shape)
            sampled_indices = np.random.randint(0, row_count, size=row_count)
            for column_index in range(column_count):
                sampled_x[:, column_index] = np.array([dataX[row_index, column_index] for row_index in sampled_indices])
            sampled_y = np.array([dataY[row_index] for row_index in sampled_indices])
            l.addEvidence(sampled_x, sampled_y)
            count += 1

    def query(self, points):
        result_y = np.zeros(points.shape[0])
        for learner in self.learners:
            temp_result = learner.query(points)
            result_y = result_y + temp_result
        result_y /= len(self.learners)
        return result_y


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """

    normed = prices / prices.ix[0, :]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)

    return port_val


def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    df = df / df.ix[0, :]
    plot_data(df, title, xlabel, ylabel)


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plot.show()


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # TODO: Your code here

    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret.ix[0] = 0
    daily_ret = daily_ret[1:]
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    cum_ret = (port_val[-1] / port_val[0]) - 1
    sharpe_ratio = math.sqrt(samples_per_year) * ((avg_daily_ret - daily_rf) / std_daily_ret)

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """

    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret.ix[0] = 0
    daily_ret = daily_ret[1:]
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    cum_ret = (port_val[-1] / port_val[0]) - 1
    sharpe_ratio = math.sqrt(samples_per_year) * ((avg_daily_ret - daily_rf) / std_daily_ret)

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


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


def test_run():
    """Driver function."""
    # Define input parameters
    # stock_list = ['ML4T-399', 'IBM']
    k_size = 2
    bag_size = 20
    window_size = 10
    stock_list = ['IBM']
    for k in range(len(stock_list)):
        print stock_list[k]
        for j in range(2):
            if j == 0:
                start_date = '2007-12-3'
                end_date = '2010-01-8'
            else:
                start_date = '2009-12-3'
                end_date = '2011-01-8'
            stock = stock_list[k]
            # Simulate a $SPX-only reference portfolio to get stats
            data = get_data([stock], pd.date_range(start_date, end_date))
            data = data[[stock]]  # remove SPY by choosing IBM
            data['SMA'] = pd.rolling_mean(data[stock], window=window_size)
            data['STD'] = pd.rolling_std(data[stock], window=window_size)
            data['HigherBand'] = data['SMA'] + (2 * data['STD'])
            data['LowerBand'] = data['SMA'] - (2 * data['STD'])
            data['BB'] = (data[stock] - data['SMA'])/(2*data['STD'])
            data['Momentum'] = (data[stock]/data[stock].shift(5)) - 1.0
            data['DailyReturns'] = (data[stock]/data[stock].shift(1)) - 1.0
            data['Volatility'] = pd.rolling_std(data['DailyReturns'], window=window_size)
            data['Y'] = (data[stock].shift(-5)/data[stock]) - 1.0
            del data['SMA']
            del data['STD']
            del data['HigherBand']
            del data['LowerBand']
            del data['DailyReturns']
            data = data.ix[20:len(data)-5]
            pd.set_option('display.max_rows', len(data))
            learner_data = data.ix[:, 1:]

            if j == 0:
                train_x = learner_data.ix[:, 0:-1]
                train_y = learner_data.ix[:, -1]
                train_x = train_x.as_matrix()
                train_y = train_y.as_matrix()
            else:
                test_x = learner_data.ix[:, 0:-1]
                test_y = learner_data.ix[:, -1]
                test_x = test_x.as_matrix()
                test_y = test_y.as_matrix()

        total = 2

        for i in range(total):
            if i == 0:
                print "LinRegLearner"
                learner = LinRegLearner()  # create a LinRegLearner
            else:
                print "BagLearner"
                learner = BagLearner(learner=KNNLearner, kwargs={"k": k_size}, bags=bag_size, boost=False)  # create a BagLearner

            learner.addEvidence(train_x, train_y)  # train it

            # evaluate in sample
            predY = learner.query(train_x)  # get the predictions
            rmse = math.sqrt(((train_y - predY) ** 2).sum()/train_y.shape[0])
            print
            print "In sample results"
            print "RMSE: ", rmse
            c = np.corrcoef(predY, y=train_y)
            print "corr: ", c[0, 1]

            # # evaluate out of sample
            predY = learner.query(test_x)  # get the predictions
            rmse = math.sqrt(((test_y - predY) ** 2).sum()/test_y.shape[0])
            print
            print "Out of sample results"
            print "RMSE: ", rmse
            c = np.corrcoef(predY, y=test_y)
            print "corr: ", c[0, 1]
            print

    # plot.figure()
    # ax = plot.gca()
    # ax.set_title('Double Bollinger Bands')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # data['IBM'].plot(label='IBM', ax=ax, color='b')
    # data['SMA'].plot(label='SMA', ax=ax, color='y')
    # data['HigherBand'].plot(label='Bollinger Bands for 2 SD', ax=ax, color='cyan')
    # data['LowerBand'].plot(label='', ax=ax, color='cyan')
    # data['MiddleHigherBand'].plot(label='Bollinger Bands for 1 SD', ax=ax, color='magenta')
    # data['MiddleLowerBand'].plot(label='', ax=ax, color='magenta')
    # ax.legend(loc='best')
    #
    # count = 0
    # line_count = 0
    # short_flag = False
    # long_flag = False
    #
    # data_array = [('Date', 'Symbol', 'Order', 'Shares')]
    #
    # for index, row in data.iterrows():
    #     current_price = data.loc[index, 'IBM']
    #     higher_2 = data.loc[index, 'HigherBand']
    #     lower_2 = data.loc[index, 'LowerBand']
    #     current_val = data.loc[index, 'PercentageMiddle']
    #
    #     if count == 0:
    #         last = current_val
    #
    #     if not short_flag and not long_flag:
    #         if last > -1.0 >= current_val:
    #             # print "short entry"
    #             line_count += 1
    #             plot.axvline(index, color='red')
    #             short_flag = True
    #             data_array.append((str(index.strftime('%Y-%m-%d')), 'IBM', 'SELL', '100'))
    #
    #         if last < 1.0 <= current_val:
    #             # print "long entry"
    #             line_count += 1
    #             plot.axvline(index, color='green')
    #             long_flag = True
    #             data_array.append((str(index.strftime('%Y-%m-%d')), 'IBM', 'BUY', '100'))
    #
    #     if short_flag or long_flag:
    #         if short_flag:
    #             if current_price <= lower_2:
    #                 # exit for short entry
    #                 line_count += 1
    #                 plot.axvline(index, color='black')
    #                 short_flag = False
    #                 long_flag = False
    #                 data_array.append((str(index.strftime('%Y-%m-%d')), 'IBM', 'BUY', '100'))
    #         if long_flag:
    #             if current_price >= higher_2:
    #                 # exit for long entry
    #                 line_count += 1
    #                 plot.axvline(index, color='black')
    #                 short_flag = False
    #                 long_flag = False
    #                 data_array.append((str(index.strftime('%Y-%m-%d')), 'IBM', 'SELL', '100'))
    #     count += 1
    #     last = current_val
    #
    # with open('orders.csv', 'w') as fp:
    #     data_writer = csv.writer(fp, delimiter=',')
    #     data_writer.writerows(data_array)
    #
    # plot.show()
    #
    # orders_file = os.path.join("", "orders.csv")
    # start_val = 10000
    # # Process orders
    # portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    #
    # # Get portfolio stats
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
    #
    # # Simulate a $SPX-only reference portfolio to get stats
    # prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    # prices_SPY = prices_SPY[['SPY']]
    # portvals_SPY = get_portfolio_value(prices_SPY, [1.0])
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(portvals_SPY)
    #
    # # Compare portfolio against $SPX
    # print "Data Range: {} to {}".format(start_date, end_date)
    # print
    # print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPY)
    # print
    # print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY: {}".format(cum_ret_SPY)
    # print
    # print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY: {}".format(std_daily_ret_SPY)
    # print
    # print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPY)
    # print
    # print "Final Portfolio Value: {}".format(portvals[-1])
    #
    # # Plot computed daily portfolio value
    # df_temp = pd.concat([portvals, prices_SPY['SPY']], keys=['Portfolio', 'SPY'], axis=1)
    # plot_normalized_data(df_temp, title="Daily portfolio value and SPY")


if __name__ == "__main__":
    test_run()
