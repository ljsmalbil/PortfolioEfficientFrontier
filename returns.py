"""
Developed by L. Smalbil
This is an object that builds the CAPM and computes the expected return for a given security
"""

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model

class Returns:
    def __init__(self, security_name, market_name = 'data/SPY'):
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

        return beta

    def expected_return(self):

        """
        One of the terms in the CAPM model is the risk-free rate.
        Here the latest 13-week US treasury bill interest rate is used as a proxy
        :return:
        """
        # Retrieve beta for asset
        returns = Returns(security_name=self.security_name)
        beta = returns.beta()

        # Retrieve the current RFR
        risk_free_rate = yf.Ticker("^IRX")
        risk_free_rate = risk_free_rate.history(period="today")
        risk_free_rate = float(risk_free_rate['Close']) / 100

        # N.B. This is a market return estimate
        estimated_market_return = 0.071
        asset_expected_return = risk_free_rate + (beta * (estimated_market_return - risk_free_rate))

        return asset_expected_return

    def scatter_returns(self, security_name = 'data/MSFT'):

        returns = Returns(security_name=self.security_name)
        data = returns.data_builder()

        x = self.market_name
        y = self.security_name

        sns.scatterplot(data=data, x=x, y=y)
        plt.title('Returns ' + self.security_name + ' and ' + self.market_name)
        plt.plot([data[self.market_name].min(), data[self.market_name].max()], [0, 0], '--', linewidth=2, color="r")
        plt.plot([0, 0], [data[self.security_name].min(), data[self.security_name].max()], '--', linewidth=2, color="r")

        return plt.show()

