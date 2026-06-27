import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import COLUMNS_TO_DROP, ONE_HOT_ENCODE_COLUMNS

logger = logging.getLogger(__name__)


class Transformer:
    def __init__(self):
        self.drop_columns = COLUMNS_TO_DROP
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS
        self.encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data transformation")
        df = df.drop(self.drop_columns, axis=1, errors="ignore")
        df = self._add_has_balance(df)
        df = self._remove_nulls(df)
        df = self._treat_outliers(df)
        df = self._one_hot_encoding(df)
        logger.info(f"Transformation complete. Output shape: {df.shape}")
        return df

    def _add_has_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["HasBalance"] = (df["Balance"] > 0).astype(int)
        logger.debug("HasBalance feature added")
        return df

    def _remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.debug(f"Removed {before - len(df)} rows with nulls")
        return df

    def _treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        age_cap = df["Age"].quantile(0.99)
        df["Age"] = df["Age"].clip(upper=age_cap)

        Q1 = df["CreditScore"].quantile(0.10)
        Q3 = df["CreditScore"].quantile(0.90)
        IQR = Q3 - Q1
        df["CreditScore"] = df["CreditScore"].clip(
            lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR
        )
        logger.debug("Outlier treatment applied to Age and CreditScore")
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        self.encoder.fit(df[self.one_hot_encoding_columns])
        encoded_df = self.encoder.transform(df[self.one_hot_encoding_columns])
        df = df.drop(columns=self.one_hot_encoding_columns)
        df = pd.concat([df, encoded_df], axis=1)
        logger.debug(f"One-hot encoding applied to: {self.one_hot_encoding_columns}")
        return df
