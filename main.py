# https://textbooks.math.gatech.edu/ila/matrix-equations.html#:~:text=cu%20)%3D%20cAu-,Definition,.%2C%20x%20n%20are%20unknown.

import numpy as np
from sklearn import linear_model
from returns import Returns

if __name__ == "__main__":

    w_a = 0.6
    w_b = 0.4

    r_a = 0.2
    r_b = 0.12

    return_portfolio = (w_a * r_a) + (w_b * r_b)


    security_name = 'MSFT'

    returns = Returns(security_name=security_name)

    data = returns.data_builder()
    print(data)

    print(returns.beta())



    """
    X = np.array(clean_monthly_returns[security_name])
    Y = np.array(clean_monthly_returns['SPY'])

    covariance_xy = np.mean((X - np.mean(X)) * (X - np.mean(Y)))
    print(covariance_xy / np.var(Y))


    #X = [1.1, 1.7, 2.1, 1.4, 0.2]
    #Y = [3, 4.2, 4.9, 4.1, 2.5]

    list_of_values = []

    for i in range(len(X)):
        element = (X[i] - np.mean(X)) * (Y[i] - np.mean(Y))
        list_of_values.append(element)

    summed = np.sum(list_of_values)

    covariance = summed / (len(X) - 1)


    beta = covariance / np.var(Y)
    print(beta)
    """



