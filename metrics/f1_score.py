from sklearn.metrics import f1_score as f1_score_sk


def f1_score(y_true, y_pred):
    """
    Metrics can be defined by code. Cubonacci calls these methods for evaluations.
    :param y_true: The targets from the DataLoader
    :param y_pred: The corresponding output from the trained model
    :return: Scalar metric score
    """
    return f1_score_sk(y_true, y_pred["Good_credit"], average="weighted")
