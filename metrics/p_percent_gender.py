import warnings

import numpy as np

from shared.data_slice import gender_male


# Implementation based on scikit-lego, see https://github.com/koaning/scikit-lego/blob/master/sklego/metrics.py

def p_percent_gender(X, y_true, y_pred):
    gender_male_values = gender_male(X)
    y_given_z1 = y_pred.loc[gender_male_values, "Good_credit"]
    y_given_z0 = y_pred.loc[~gender_male_values, "Good_credit"]
    p_y1_z1 = np.mean(y_given_z1 == 1.0)
    p_y1_z0 = np.mean(y_given_z0 == 1.0)

    if p_y1_z1 == 0:
        warnings.warn(
            f"No samples with y_pred['Good_credit'] == 1 for male gender, returning 0",
            RuntimeWarning,
        )
        return 0

    if p_y1_z0 == 0:
        warnings.warn(
            f"No samples with y_pred['Good_credit'] == 1 for female gender, returning 0",
            RuntimeWarning,
        )
        return 0

    p_percent = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
    return p_percent if not np.isnan(p_percent) else 1
