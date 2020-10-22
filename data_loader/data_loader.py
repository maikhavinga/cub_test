import pandas as pd
from .utils import transform_target, categorize_data, int_to_float_data


class DataLoader:
    def load_training_data(self):
        """
        Method that returns two Python objects, the features and the corresponding targets. This will usually involve
        either querying a database or downloading one or more files from the internet. When credentials are needed,
        these can be added as "Secrets" in the UI. By adding `secrets` as argument to the `load_data` method, Cubonacci
        knows to inject the secrets in the method. The `secrets` argument is a dictionary where the keys are the names
        of the secrets and the values are the values given in the user interface.
        :return: (features, target) tuple. Both the features and the target are Python objects that need to be iterable
                 and need to be the same size. Cubonacci automatically infers the relevant schemas. Column names or
                 dictionary keys can only contain letters and underscores at the moment.
        """
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

        data = pd.read_csv(url, delimiter=' ', header=None)
        columns = ['Status_current_account',
                   'Duration',
                   'Credit_history',
                   'Purpose',
                   'Credit_amount',
                   'Savings',
                   'Employment_status',
                   'Installment_rate',
                   'Personal_status',
                   'Other_debtors',
                   'Residence_since',
                   'Property',
                   'Age_years',
                   'Other_installment_plans',
                   'Housing',
                   'Number_existing_credits',
                   'Job',
                   'Number_liable_people',
                   'Telephone',
                   'Foreign_worker',
                   'Good_credit']
        data.columns = columns

        data = categorize_data(data)
        data = int_to_float_data(data)
        data = transform_target(data)

        features = data[[c for c in data.columns if c != 'Good_credit']]
        target = data['Good_credit']

        print(f"Data loader retrieved {len(features)} samples")

        return features, target