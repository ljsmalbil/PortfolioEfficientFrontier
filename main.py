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

    # Path to folder 
    security_name = 'data/MSFT'

    returns = Returns(security_name=security_name)

    data = returns.data_builder()
    print(returns.beta())

    print(returns.expected_return())




