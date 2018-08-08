# Utility function for calculating a and b for linear regression


def calculate_y_hat(X, Y):
    # Calculate the best fit a + b
    denominator = X.dot(X) - X.mean() * X.sum()

    a = (Y.dot(X).sum() - Y.mean() * X.sum()) / denominator
    b = (Y.mean() * X.dot(X) - X.mean() * Y.dot(X).sum()) / denominator

    return a * X + b


def calculate_r_squared(Y, Y_hat):
    residual = Y - Y_hat
    total = Y - Y.mean()
    r_squared = 1 - residual.dot(residual).sum() \
                / total.dot(total).sum()
    return r_squared

