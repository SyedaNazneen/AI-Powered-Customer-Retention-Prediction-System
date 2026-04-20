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

# Machine Learning Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

# Logger setup
logger = setup_logging("all_models")


def knn(X_train, y_train, X_test, y_test):
    global knn_predictions
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    knn_predictions = model.predict(X_test)
    logger.info(f"KNN Accuracy: {accuracy_score(y_test, knn_predictions)}")


def nb(X_train, y_train, X_test, y_test):
    global nb_predictions
    model = GaussianNB()
    model.fit(X_train, y_train)
    nb_predictions = model.predict(X_test)
    logger.info(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_predictions)}")


def lr(X_train, y_train, X_test, y_test):
    global lr_predictions
    model = LogisticRegression()
    model.fit(X_train, y_train)
    lr_predictions = model.predict(X_test)
    logger.info(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions)}")


def dt(X_train, y_train, X_test, y_test):
    global dt_predictions
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    dt_predictions = model.predict(X_test)
    logger.info(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_predictions)}")


def rf(X_train, y_train, X_test, y_test):
    global rf_predictions
    model = RandomForestClassifier(criterion='entropy', n_estimators=5)
    model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)
    logger.info(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}")


def adab(X_train, y_train, X_test, y_test):
    global ada_predictions
    base_lr = LogisticRegression()
    model = AdaBoostClassifier(estimator=base_lr, n_estimators=5)
    model.fit(X_train, y_train)
    ada_predictions = model.predict(X_test)
    logger.info(f"AdaBoost Accuracy: {accuracy_score(y_test, ada_predictions)}")


def gb(X_train, y_train, X_test, y_test):
    global gb_predictions
    model = GradientBoostingClassifier(n_estimators=5)
    model.fit(X_train, y_train)
    gb_predictions = model.predict(X_test)
    logger.info(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_predictions)}")


def xgb_model(X_train, y_train, X_test, y_test):
    global xgb_predictions
    model = XGBClassifier(n_estimators=5)
    model.fit(X_train, y_train)
    xgb_predictions = model.predict(X_test)
    logger.info(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_predictions)}")


def auc_roc_tech(y_test):
    """Sare models ki performance plot karna"""
    plt.figure(figsize=(8, 6))
    plt.plot([5], "k--")

    # ROC Curves calculate karna
    models = {
        'KNN': knn_predictions, 'LR': lr_predictions,
        'NB': nb_predictions, 'DT': dt_predictions,
        'RF': rf_predictions, 'Ada': ada_predictions,
        'GB': gb_predictions, 'XGB': xgb_predictions
    }

    for name, preds in models.items():
        fpr, tpr, _ = roc_curve(y_test, preds)
        plt.plot(fpr, tpr, label=name)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ALL Models AUC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()


def common(X_train, y_train, X_test, y_test):
    """Pipeline se calls receive karke saare models execute karna"""
    try:
        logger.info('-----------knn----------------')
        knn(X_train, y_train, X_test, y_test)
        logger.info('-----------nb----------------')
        nb(X_train, y_train, X_test, y_test)
        logger.info('-----------lr----------------')
        lr(X_train, y_train, X_test, y_test)
        logger.info('-----------dt----------------')
        dt(X_train, y_train, X_test, y_test)
        logger.info('-----------rf----------------')
        rf(X_train, y_train, X_test, y_test)
        logger.info('-----------adab----------------')
        adab(X_train, y_train, X_test, y_test)
        logger.info('-----------gb----------------')
        gb(X_train, y_train, X_test, y_test)
        logger.info('-----------xgb_model----------------')
        xgb_model(X_train, y_train, X_test, y_test)

        logger.info("Generating AUC-ROC Chart...")
        auc_roc_tech(y_test)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in all_models at line {er_line.tb_lineno}: {er_msg}")