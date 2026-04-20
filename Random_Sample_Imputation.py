import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
from logging_code import setup_logging

# Logger setup karna specifically is file ke liye
logger = setup_logging("Random_Sample_Imputation")


def handle_missing_value(X_train, X_test):
    """X_train aur X_test mein missing values ko random sampling se fill karna"""
    try:
        # Imputation se pehle ki info log karna
        logger.info(f"Before Handling Null values X_train shape: {X_train.shape} \nNulls: \n{X_train.isnull().sum()}")
        logger.info(f"Before Handling Null values X_test shape: {X_test.shape} \nNulls: \n{X_test.isnull().sum()}")

        for i in X_train.columns:
            # Agar kisi column mein null values hain
            if X_train[i].isnull().sum() > 0:
                # Naya column banana replaced values ke liye
                X_train[i + "_replaced"] = X_train[i].copy()
                X_test[i + "_replaced"] = X_test[i].copy()

                # Random sample nikalna (dropna karke taake sirf valid values milen)
                s = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
                s1 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)

                # Sample ke index ko missing values ke index se match karna
                s.index = X_train[X_train[i].isnull()].index
                s1.index = X_test[X_test[i].isnull()].index

                # Missing spots par sample values fill karna
                X_train.loc[X_train[i].isnull(), i + "_replaced"] = s
                X_test.loc[X_test[i].isnull(), i + "_replaced"] = s1

                # Purana column drop kardein jisme null values thi
                X_train = X_train.drop([i], axis=1)
                X_test = X_test.drop([i], axis=1)

        # Imputation ke baad ki info log karna
        logger.info(f"After Handling Null values X_train shape: {X_train.shape} \nNulls: \n{X_train.isnull().sum()}")
        logger.info(f"After Handling Null values X_test shape: {X_test.shape} \nNulls: \n{X_test.isnull().sum()}")

        return X_train, X_test

    except Exception as e:
        # Error trace karna line number ke saath
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in handle_missing_value at line {er_line.tb_lineno}: {er_msg}")