import warnings

import numpy as np

from shared.data_slice import gender_male


# Implementation based on scikit-lego, see https://github.com/koaning/scikit-lego/blob/master/sklego/metrics.py

def equal_opportunity_gender(X, y_true, y_pred):
    gender_male_values = gender_male(X)
    y_given_z1_y1 = y_pred.loc[gender_male_values & y_true, "Good_credit"]
    y_given_z0_y1 = y_pred.loc[~gender_male_values & y_true, "Good_credit"]

    # If we never predict a positive target for one of the subgroups, the model is by definition not
    # fair so we return 0
    if len(y_given_z1_y1) == 0:
        warnings.warn(
            f"No samples with y_pred['Good_credit'] == 1 for male gender and y_true['Good_credit'], returning 0",
            RuntimeWarning,
        )
        return 0

    if len(y_given_z0_y1) == 0:
        warnings.warn(
            f"No samples with y_pred['Good_credit'] == 1 for female gender and y_true['Good_credit'], returning 0",
            RuntimeWarning,
        )
        return 0

    p_y1_z1 = np.mean(y_given_z1_y1 == 1.0)
    p_y1_z0 = np.mean(y_given_z0_y1 == 1.0)
    score = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
    return score if not np.isnan(score) else 1
