


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import pickle

class LoanModel:
    def __init__(self):

        # Encoder
        self.ordinal_columns = ['person_education']
        self.one_hot_columns = ['person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file', 'person_gender']
        self.education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']

        self.ordinal_encoder = OrdinalEncoder(categories=[self.education_order])
        self.onehot_encoder = OneHotEncoder()

        # Preprocessor columnTransformer
        self.preprocessor = ColumnTransformer(transformers=[
            ('ordinal', self.ordinal_encoder, self.ordinal_columns),
            ('onehot', self.onehot_encoder, self.one_hot_columns)
        ])

        # Model xgboost
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)

        self.pipeline = Pipeline(steps=[
            ('preprocessing', self.preprocessor),
            ('classifier', self.model)
        ])

    def _clean_data(self, data):
        # benerin gender
        data['person_gender'] = data['person_gender'].str.lower().str.strip()
        data['person_gender'] = data['person_gender'].replace({
        'fe male': 'female',
        'male': 'male',
        'Male': 'male' })

        # Imputasi income berdasarkan median education
        income_medians = data.groupby('person_education')['person_income'].median()
        data['person_income'] = data.apply(lambda row: income_medians.get(row['person_education'], data['person_income'].median())
                                           if pd.isna(row['person_income']) else row['person_income'], axis=1)

        # Log transform
        data['person_income'] = np.log1p(data['person_income'])

        # Drop age outlier > 90
        data = data[data['person_age'] <= 90]
        return data

    def fit(self, data):
        data = self._clean_data(data.copy())
        X = data.drop(columns=['loan_status'])
        y = data['loan_status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def predict(self, data):
        data = self._clean_data(data.copy())
        return self.pipeline.predict(data)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {filepath}")

data = pd.read_csv("Dataset_A_loan.csv")  
loan_model = LoanModel()
loan_model.fit(data)

loan_model.save_model('xgboost_full_pipeline.pkl')
