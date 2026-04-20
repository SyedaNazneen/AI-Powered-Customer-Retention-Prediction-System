import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
import sys
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # <--- YE IMPORT ZAROORI HAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from all_models import common

logger = setup_logging("feature_scaling")


def fs(X_train, y_train, X_test, y_test):
    try:
        # 1. Data Scaling
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        with open('standar_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)

        logger.info("✔ Feature Scaling completed and Scaler saved.")

        # 2. Multi-Model Comparison (Scaled data pass karein)
        common(X_train_sc, y_train, X_test_sc, y_test)

        # 3. Best Model Building (XGBoost)
        # Humne baseline ke bajaye best performer (XGBoost) select kiya hai
        reg = XGBClassifier(n_estimators=5)
        reg.fit(X_train_sc, y_train)

        # 4. Model Evaluation (y_pred variable use karein)
        y_pred = reg.predict(X_test_sc)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Test Accuracy : {accuracy}")
        logger.info(f"Test Confusion Matrix : \n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification report : \n{classification_report(y_test, y_pred)}")

        # Trained model ko save karna
        with open('Model.pkl', 'wb') as t:
            pickle.dump(reg, t)

        logger.info("✔ XGBoost model trained and saved as Model.pkl.")
        print(f"Training Complete. XGBoost Accuracy: {accuracy:.2f}")

        return X_train_sc, X_test_sc

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")