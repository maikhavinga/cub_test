from sklearn.metrics import accuracy_score


def accuracy(y_true, y_pred):
    """
    Metrics can be defined by code. Cubonacci calls these methods for evaluations.
    :param y_true: The targets from the DataLoader
    :param y_pred: The corresponding output from the trained model
    :return: Scalar metric score
    """
    return accuracy_score(y_true=y_true, y_pred=y_pred["Good_credit"])
