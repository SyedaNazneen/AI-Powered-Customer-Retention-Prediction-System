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
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

# Logger setup
logger = setup_logging("filter_methods")


def fm(X_train_num, X_test_num, y_train, y_test):
    """Numerical features select karna Variance aur Correlation ke zariye [1]"""
    try:
        logger.info(f"Before Train Columns : {X_train_num.shape} \n : {X_train_num.columns}")
        logger.info(f"Before Test Columns : {X_test_num.shape} \n : {X_test_num.columns}")

        # 1. Variance Threshold (Constant features hatana) [2]
        reg = VarianceThreshold(threshold=0.01)
        reg.fit(X_train_num)

        logger.info(f"Number of Good Columns : {sum(reg.get_support())} : {X_train_num.columns[reg.get_support()]}")
        logger.info(f"Number of Bad Columns : {sum(~reg.get_support())} : {X_train_num.columns[~reg.get_support()]}")

        # 2. Hypothesis Testing (Pearson Correlation) [3]
        logger.info("====================Hypothesis Testing=================================")
        c = []
        for i in X_train_num.columns:
            results = pearsonr(X_train_num[i], y_train)  # [4]
            c.append(results)

        t = np.array(c)
        p_value = pd.Series(t[:, 1], index=X_train_num.columns)
        logger.info(f"P-Values: \n{p_value}")

        # Final logging and return [5]
        logger.info(f"After Train Columns : {X_train_num.shape}")
        logger.info(f"After Test Columns : {X_test_num.shape}")
        return X_train_num, X_test_num

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")