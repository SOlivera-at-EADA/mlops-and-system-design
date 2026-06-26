import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import COLUMNS_TO_DROP, ONE_HOT_ENCODE_COLUMNS


class Transformer:
    def __init__(self):
        self.drop_columns = COLUMNS_TO_DROP
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS
        self.encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(self.drop_columns, axis=1, errors="ignore")
        df = self._add_has_balance(df)
        df = self._remove_nulls(df)
        df = self._treat_outliers(df)
        df = self._one_hot_encoding(df)
        return df

    def _add_has_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["HasBalance"] = (df["Balance"] > 0).astype(int)
        return df

    def _remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna().reset_index(drop=True)

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
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        self.encoder.fit(df[self.one_hot_encoding_columns])
        encoded_df = self.encoder.transform(df[self.one_hot_encoding_columns])
        df = df.drop(columns=self.one_hot_encoding_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df
