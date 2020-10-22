from sklearn.metrics import log_loss as log_loss_sk


def log_loss(y_true, y_pred):
    """
    Metrics can be defined by code. Cubonacci calls these methods for evaluations.
    :param y_true: The targets from the DataLoader
    :param y_pred: The corresponding output from the trained model
    :return: Scalar metric score
    """
    return log_loss_sk(y_true, y_pred[['Probability_no', 'Probability_yes']])
