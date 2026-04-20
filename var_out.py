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
from scipy.stats import yeojohnson

# Logger setup karna specifically is module ke liye
logger = setup_logging("var_out")


def vt_outliers(X_train_num, X_test_num):
    """Numerical columns par Yeo-Johnson transformation aur IQR trimming apply karna"""
    try:
        # Transformation se pehle column names log karein
        logger.info(f"Before Transformation Train Columns: {X_train_num.columns.tolist()}")
        logger.info(f"Before Transformation Test Columns: {X_test_num.columns.tolist()}")

        for i in X_train_num.columns:
            # 1. Yeo-Johnson Transformation (Data ko normal distribution ke kareeb lane ke liye) [1]
            X_train_num[i + '_yeo'], lam_value = yeojohnson(X_train_num[i])
            # Test data par bhi wahi transformation apply karein [2]
            X_test_num[i + '_yeo'], _ = yeojohnson(X_test_num[i])

            # Purane original columns drop kardein
            X_train_num = X_train_num.drop([i], axis=1)
            X_test_num = X_test_num.drop([i], axis=1)

            # 2. Outlier Trimming (IQR Method) [2]
            # Quartiles aur IQR calculate karein (Sirf Train data se limits nikalni hain)
            q3 = X_train_num[i + '_yeo'].quantile(0.75)
            q1 = X_train_num[i + '_yeo'].quantile(0.25)
            iqr = q3 - q1

            upper_limit = q3 + (1.5 * iqr)
            lower_limit = q1 - (1.5 * iqr)

            # np.where use karke values ko limits ke andar cap (trim) karein [2, 3]
            X_train_num[i + '_trim'] = np.where(X_train_num[i + '_yeo'] > upper_limit, upper_limit,
                                                np.where(X_train_num[i + '_yeo'] < lower_limit, lower_limit,
                                                         X_train_num[i + '_yeo']))

            X_test_num[i + '_trim'] = np.where(X_test_num[i + '_yeo'] > upper_limit, upper_limit,
                                               np.where(X_test_num[i + '_yeo'] < lower_limit, lower_limit,
                                                        X_test_num[i + '_yeo']))

            # Intermediate '_yeo' columns ko drop karke sirf '_trim' wale rakhein [3]
            X_train_num = X_train_num.drop([i + '_yeo'], axis=1)
            X_test_num = X_test_num.drop([i + '_yeo'], axis=1)

        # Transformation ke baad ki info log karein [4]
        logger.info(f"After Transformation Train Columns: {X_train_num.columns.tolist()}")
        logger.info(f"After Transformation Test Columns: {X_test_num.columns.tolist()}")

        return X_train_num, X_test_num

    except Exception as e:
        # Error trace karna line number ke saath
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in vt_outliers at line {er_line.tb_lineno}: {er_msg}")