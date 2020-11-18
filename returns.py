"""
Developed by L. Smalbil
This is an object that builds the CAPM and computes the expected return for a given security
"""

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn import linear_model

class Returns:

    def __init__(self, security_name, market_name = 'SPY'):
        self.security_name = security_name
        self.market_name = market_name

    def data_builder(self):
        market = pd.read_csv(self.market_name +'.csv')
        security = pd.read_csv(self.security_name + '.csv')

        # Join the two dfs
        monthly_prices = pd.concat([security['Close'], market['Close']], axis=1)
        monthly_prices.columns = [self.security_name, self.market_name]

        # Convert to percentages
        monthly_returns = monthly_prices.pct_change(1)

        # Drop in case values are missing
        clean_monthly_returns = monthly_returns.dropna(axis=0)

        return clean_monthly_returns

    def beta(self):
        returns = Returns(security_name=self.security_name)

        data = returns.data_builder()

        X = np.array(data[self.security_name])
        Y = np.array(data[self.market_name])

        covariance_xy = (X - np.mean(X)) * (Y - np.mean(Y))
        covariance_xy = np.mean(covariance_xy)
        beta = covariance_xy / np.var(Y)

        """
        list_of_values = []

        for i in range(len(X)):
            element = (X[i] - np.mean(X)) * (Y[i] - np.mean(Y))
            list_of_values.append(element)

        summed = np.sum(list_of_values)

        covariance = summed / (len(X) - 1)
        beta = covariance / np.var(Y)
        """

        return beta


