# Utility function for calculating a and b for linear regression

def calculate_linear_values(X, Y):
    # Calculate the best fit a + b
    denominator = X.dot(X) - X.mean() * X.sum()

    a = (Y.dot(X).sum() - Y.mean() * X.sum()) / denominator
    b = (Y.mean() * X.dot(X) - X.mean() * Y.dot(X).sum()) / denominator

    return a * X + b