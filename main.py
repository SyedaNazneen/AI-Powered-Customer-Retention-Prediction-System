import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import os
import seaborn as sns

import logging
from logging_code import setup_logging

logger = setup_logging("retention_main")
import sys
from sklearn.model_selection import train_test_split

from Random_Sample_Imputation import handle_missing_value
from filter_methods import fm
from categorical_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs

class RETENTION:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)  # Loading dataset
            logger.info(f"Total data_size : {self.df.shape}")

            # customerID drop karna kyunki ye prediction mein useful nahi hai [4]
            if 'customerID' in self.df.columns:
                self.df = self.df.drop(['customerID'], axis=1)

            logger.info(f"Null values check: \n{self.df.isnull().sum()}")

            # Independent (X) aur Dependent (y) variables split karna
            self.X = self.df.iloc[:, :-1]  # Features (including your 'sim' column)
            self.y = self.df.iloc[:, -1]  # Target (Churn)

            # Train-Test Split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Target variable 'Churn' ko numeric map karna (Yes=1, No=0) [4, 5]
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)

            logger.info(f"Train Data Size : {self.X_train.shape}")
            logger.info(f"Test Data Size : {self.X_test.shape}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in __init__ at line {er_line.tb_lineno}: {er_msg}")

    def preprocessing(self):
        """TotalCharges conversion aur missing values handle karna [4, 6]"""
        try:
            # TotalCharges ko numeric banana
            '''Constant Value Imputation: Iska matlab hai ke jahan bhi data missing (NaN) tha, humne use ek fixed number
            yaani 0 se bhar diya hai 
            TotalCharges ko string se numeric mein convert karte waqt (errors='coerce'), jo values sahi convert nahi ho pateen wo empty ho jati hain, 
            isliye unhe 0 se fill karna zaroori hota hai'''
            self.X_train['TotalCharges'] = pd.to_numeric(self.X_train['TotalCharges'], errors='coerce')
            self.X_test['TotalCharges'] = pd.to_numeric(self.X_test['TotalCharges'], errors='coerce')

            # Null values ko 0 se fill karna (ya helper function use karein) [7]
            self.X_train['TotalCharges'] = self.X_train['TotalCharges'].fillna(0)
            self.X_test['TotalCharges'] = self.X_test['TotalCharges'].fillna(0)

            logger.info("Data Cleaning and Numeric Conversion complete.")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in preprocessing at line {er_line.tb_lineno}: {er_msg}")

    def data_seperation(self):
        """Numeric aur Categorical columns ko alag karna [8]"""
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')

            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')

            logger.info(f"Separated Numerical Columns: {self.X_train_num_cols.columns.tolist()}")
            logger.info(f"Separated Categorical Columns: {self.X_train_cat_cols.columns.tolist()}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in data_seperation at line {er_line.tb_lineno}: {er_msg}")

    def feature_selection(self):  # [7]
        try:
            # Numerical features ko filter methods ke zariye select karna
            self.X_train_num_cols, self.X_test_num_cols = fm(
                self.X_train_num_cols, self.X_test_num_cols, self.y_train, self.y_test
            )
            logger.info("Feature Selection stage completed successfully.")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def cat_to_num(self):
        """Categorical data ko numerical mein badalna aur numerical data ke saath merge karna"""
        try:
            logger.info("Starting Categorical to Numerical conversion...")

            # 1. External module 'c_t_n' ko call karna jo One-Hot aur Ordinal encoding apply karta hai
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)

            # 2. Index reset karna taake concatenation ke waqt rows upar neeche na hon
            # Sources ke mutabiq concatenation se pehle index reset zaroori hai
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)

            # 3. Numerical aur processed Categorical columns ko merge (concat) karna
            # 'sim' column ab numerical features mein convert ho chuka hoga
            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)

            # 4. Final Data Monitoring
            logger.info(f"==================== Final Stage 1 Output ====================")
            logger.info(f"Final Training data shape : {self.training_data.shape}")
            logger.info(f"Final Testing data shape : {self.testing_data.shape}")
            logger.info(f"Final Columns after Encoding: {self.training_data.columns.tolist()}")
            logger.info(f"Null values check in final data: \n{self.training_data.isnull().sum()}")

            logger.info("Encoding and Data Merging complete.")

        except Exception as e:
            # Industrial error tracing line number ke saath [1]
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in cat_to_num at line {er_line.tb_lineno}: {er_msg}")

    def data_balancing(self):
        """SMOTE apply karke imbalanced data ko handle karna"""
        try:
            # 1. Resampling se pehle data ki halat log karna
            # Mapping ke mutabiq: Good=1 (Retained), Bad=0 (Churned)
            logger.info(f"Number of Rows for Good Customer (1) before SMOTE: {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for Bad Customer (0) before SMOTE: {sum(self.y_train == 0)}")
            logger.info(f"Initial Training data shape: {self.training_data.shape}")

            # 2. SMOTE Object banana [1]
            # random_state=42 isliye use kiya gaya hai taake results consistent rahein
            sm = SMOTE(random_state=42)

            # 3. Data ko fit aur resample karna
            # Ye minority class ke liye synthetic samples create karega
            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            # 4. Resampling ke baad results log karna [2]
            logger.info(f"Number of Rows for Good Customer (1) after SMOTE: {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad Customer (0) after SMOTE: {sum(self.y_train_bal == 0)}")
            logger.info(f"Balanced Training data shape: {self.training_data_bal.shape}")

            # 5. Next Stage (Feature Scaling) ko call karna
            # Balanced data ko scaling ke liye bheja ja raha hai taake model training sahi ho
            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)

            logger.info("Data Balancing stage completed successfully.")

        except Exception as e:
            # Industrial standard error tracing
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in data_balancing at line {er_line.tb_lineno} due to: {er_msg}")


# --- Execution Block ---
if __name__ == "__main__":
    try:
        obj = RETENTION('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.preprocessing()
        obj.data_seperation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing() # SMOTE logic ke liye
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Main execution error at line {er_line.tb_lineno}: {er_msg}")