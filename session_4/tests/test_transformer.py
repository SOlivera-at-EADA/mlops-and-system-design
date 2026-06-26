import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.transform import Transformer


def test_add_has_balance_zero_balance():
    transformer = Transformer()
    df = pd.DataFrame({"Balance": [0.0, 0.0]})
    result = transformer._add_has_balance(df)
    assert list(result["HasBalance"]) == [0, 0]


def test_add_has_balance_nonzero_balance():
    transformer = Transformer()
    df = pd.DataFrame({"Balance": [0.0, 100000.0, 50000.0]})
    result = transformer._add_has_balance(df)
    assert list(result["HasBalance"]) == [0, 1, 1]


def test_remove_nulls_drops_rows():
    transformer = Transformer()
    df = pd.DataFrame({"Age": [25.0, None, 35.0], "Balance": [0.0, 100.0, 200.0]})
    result = transformer._remove_nulls(df)
    assert len(result) == 2
    assert result.isnull().sum().sum() == 0


def test_treat_outliers_winsorizes_age():
    transformer = Transformer()
    ages = [30.0] * 99 + [500.0]
    credit = [650.0] * 100
    df = pd.DataFrame({"Age": ages, "CreditScore": credit})
    result = transformer._treat_outliers(df)
    assert result["Age"].max() < 500.0


def test_treat_outliers_clips_creditscore():
    transformer = Transformer()
    scores = [650.0] * 95 + [1.0] * 3 + [900.0, 950.0]
    ages = [35.0] * 100
    df = pd.DataFrame({"Age": ages, "CreditScore": scores})
    result = transformer._treat_outliers(df)
    assert result["CreditScore"].min() >= 1.0
    assert result["CreditScore"].max() <= 950.0
