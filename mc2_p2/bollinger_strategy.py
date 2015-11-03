__author__ = 'Mowgli'

import os
import pandas as pd
import matplotlib.pyplot as plot


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


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'
    end_date = '2009-12-31'

    # Simulate a $SPX-only reference portfolio to get stats
    data = get_data(['IBM'], pd.date_range(start_date, end_date))
    data = data[['IBM']]  # remove SPY by choosing IBM
    data['SMA'] = pd.rolling_mean(data['IBM'], window=20)
    data['STD'] = pd.rolling_std(data['IBM'], window=20)
    data['HigherBand'] = data['SMA'] + 2*data['STD']
    data['LowerBand'] = data['SMA'] - 2*data['STD']
    data['Points'] = (data['IBM'] - data['SMA'])/(2*data['STD'])
    pd.set_option('display.max_rows', len(data))

    plot.figure()
    ax = plot.gca()
    data['IBM'].plot(label='IBM', ax=ax, color='b')
    data['SMA'].plot(label='SMA', ax=ax, color='y')
    data['HigherBand'].plot(label='Bollinger Bands', ax=ax, color='cyan')
    data['LowerBand'].plot(label='', ax=ax, color='cyan')
    ax.legend(loc='best')

    count = 0
    line_count = 0
    short_flag = False
    long_flag = False

    for index, row in data.iterrows():
        current = row.values[5]
        price = row.values[0]
        avg = row.values[1]

        if count == 0:
            last = row.values[5]

        if not short_flag and not long_flag:
            if last > 1.0 >= current:
                # print "short entry"
                line_count += 1
                plot.axvline(index, color='red')
                short_flag = True
            elif current >= -1.0 > last:
                # print "long entry"
                line_count += 1
                plot.axvline(index, color='green')
                long_flag = True

        if short_flag or long_flag:
            if short_flag:
                if price <= avg:
                    line_count += 1
                    plot.axvline(index, color='black')
                    short_flag = False
                    long_flag = False
            if long_flag:
                if price >= avg:
                    line_count += 1
                    plot.axvline(index, color='black')
                    short_flag = False
                    long_flag = False

        count += 1
        last = current

    plot.show()


if __name__ == "__main__":
    test_run()
