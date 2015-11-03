__author__ = 'Mowgli'

import os
import pandas as pd
import matplotlib.pyplot as plt


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2010-06-01'
    end_date = '2010-12-31'

    symbol_allocations = OrderedDict([('GOOG', 0.2), ('AAPL', 0.2), ('GLD', 0.4), ('XOM', 0.2)])  # symbols and corresponding allocations
    #symbol_allocations = OrderedDict([('AXP', 0.0), ('HPQ', 0.0), ('IBM', 0.0), ('HNZ', 1.0)])  # allocations from wiki example

    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.2, 0.4, 0.2]
    start_val = 1000000  # starting value of portfolio

    # Assess the portfolio
    assess_portfolio(start_date, end_date, symbols, allocs, start_val)


if __name__ == "__main__":
    test_run()