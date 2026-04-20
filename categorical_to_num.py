import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import logging
from logging_code import setup_logging
import sys
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Logger setup specifically for this module
logger = setup_logging("categorical_to_num")


def c_t_n(X_train_cat, X_test_cat):
    """Categorical features ko numerical mein badalna """
    try:
        logger.info(f"Before X_train_cat shape: {X_train_cat.shape}:\n:{X_train_cat.columns}")
        logger.info(f"Before X_test_cat shape: {X_train_cat.shape}:\n:{X_train_cat.columns}")

        # 1. Nominal Columns (One-Hot Encoding)
        # In features mein koi order nahi hota. 'drop=first' dummy trap se bachata hai
        target_nominal = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'PaperlessBilling', 'PaymentMethod', 'sim'
        ]

        oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        oh.fit(X_train_cat[target_nominal])

        # Transformation
        val_train = oh.transform(X_train_cat[target_nominal])
        val_test = oh.transform(X_test_cat[target_nominal])

        t1 = pd.DataFrame(val_train, columns=oh.get_feature_names_out())
        t2 = pd.DataFrame(val_test, columns=oh.get_feature_names_out())

        # Reset Index taake concatenation sahi ho [2, 3]
        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)

        X_train_cat = pd.concat([X_train_cat, t1], axis=1)
        X_test_cat = pd.concat([X_test_cat, t2], axis=1)

        # Purane nominal columns ko drop karna [3]
        X_train_cat.drop(target_nominal, axis=1, inplace=True)
        X_test_cat.drop(target_nominal, axis=1, inplace=True)

        logger.info(f"After Nominal Encoding shape: {X_train_cat.shape}")

        # 2. Ordinal Columns (Ordinal Encoding)
        # In features mein ek logical order hota hai
        target_ordinal = [
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract'
        ]

        od = OrdinalEncoder()
        od.fit(X_train_cat[target_ordinal])

        # Transformation [4]
        res_train = od.transform(X_train_cat[target_ordinal])
        res_test = od.transform(X_test_cat[target_ordinal])

        p1 = pd.DataFrame(res_train, columns=[col + "_od" for col in od.get_feature_names_out()])
        p2 = pd.DataFrame(res_test, columns=[col + "_od" for col in od.get_feature_names_out()])

        X_train_cat = pd.concat([X_train_cat, p1], axis=1)
        X_test_cat = pd.concat([X_test_cat, p2], axis=1)

        # Purane ordinal columns ko drop karna [5]
        X_train_cat.drop(target_ordinal, axis=1, inplace=True)
        X_test_cat.drop(target_ordinal, axis=1, inplace=True)

        logger.info(f"Final Categorical to Numerical shape: {X_train_cat.shape}")

        # Null values check
        logger.info(f"Train Null values check: \n{X_train_cat.isnull().sum()}")

        return X_train_cat, X_test_cat

    except Exception as e:
        # Industrial error tracing with line number
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
        raise e