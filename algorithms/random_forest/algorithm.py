from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd


class Algorithm:
    """
    Every Model class needs to implement at least the `__init__`, `fit` and `predict` methods.
    """
    def __init__(self, criterion, max_depth, max_features):
        """
        The __init__ method gets called when initializing the model. The arguments that are named in the method are
        hyperparameters that influence the way the model is constructed and trained. These can be directly linked to
        the algorithm but also for the preprocessor which is considered to be part of the model. In this example random
        forest Cubonacci can change the `criterion`, `max_depth` and `max_features` to optimize the model.
        :param criterion: Metric used for splitting heuristic
        :param max_depth: Maximum depth of individual tree
        :param max_features: Number of features to sample for individual tree
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.standard_scaler = None
        self.model = None
        self.pipeline = None

    def fit(self, X, y):
        """
        The method called to train the model. X has the same schema as the features from the DataLoader while y has the
        same schema as the target
        :param X: Features
        :param y: Target
        """
        feature_columns = ['Status_current_account',
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
                           'Foreign_worker']
        self.feature_columns = feature_columns

        numeric_features = ['Duration',
                            'Credit_amount',
                            'Installment_rate',
                            'Residence_since',
                            'Age_years',
                            'Number_existing_credits',
                            'Number_liable_people']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_features = [column for column in feature_columns if column not in numeric_features]
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        print("Instantiating model...")
        self.model = RandomForestClassifier(n_estimators=100,
                                            criterion=self.criterion,
                                            max_depth=self.max_depth,
                                            max_features=self.max_features)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('random_forest', self.model)])
        print(f"Fitting model on {len(X)} samples...")
        self.pipeline.fit(X=X, y=y.values.ravel())

        print("Calculating feature importance dictionary...")
        onehot_columns = list(self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(input_features=categorical_features))
        self.final_feature_names = numeric_features + onehot_columns
        self.feature_importance_dict = {feature: value for feature, value in zip(self.final_feature_names, self.pipeline.named_steps["random_forest"].feature_importances_)}

    def predict(self, X):
        """
        After the model has been trained Cubonacci calls this method to predict on features `X`. This method returns
        the corresponding predictions. `X` is always wrapped as if it is multiple samples, even if it is just one.
        The return value for a single prediction can be custom again. Cubonacci runs predictions on a sample of 1000
        training rows to automatically infer the target.
        :param X: Features (always multiple samples)
        :return: Prediction object
        """
        X = pd.DataFrame(X, columns=self.feature_columns)
        discrete_prediction = self.pipeline.predict(X)
        predictions = self.pipeline.predict_proba(X)
        predictions = pd.DataFrame(predictions, index=X.index, columns=['Probability_no', 'Probability_yes'])
        predictions['Good_credit'] = discrete_prediction
        return predictions

    def feature_importance(self):
        return self.feature_importance_dict
