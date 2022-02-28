# -*- coding: utf-8 -*-

"""
module: useful_functions
These are some helper functions that are used in the processing scripts
"""

import numpy as np
import datetime
# TODO: validate if these functions will work on the DL

def get_prev_months_last_date():
    """From current date returns the last date of the previous month"""
    current_date = datetime.date.today()
    first_day_of_month = current_date.replace(day=1)
    out = first_day_of_month - datetime.timedelta(days=1)
    return out


def get_prev_prev_months_last_date():
    """From current date returns the last date of the month before the previous month"""
    last_day_of_prev_month = get_prev_months_last_date()
    first_day_of_prev_month = last_day_of_prev_month.replace(day=1)
    out = first_day_of_prev_month - datetime.timedelta(days=1)
    return out


def simple_imputation(df):
    """Gets a data frame and imputes the median for numerical columns and the
    mode (most common category) for categorical columns"""
    numerical_cols = df.select_dtypes(np.number).columns
    categorical_cols = df.select_dtypes(object).columns

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df
