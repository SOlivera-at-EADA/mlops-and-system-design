import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from metadata import MODEL_PARAMS

logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, target_column: str) -> DecisionTreeClassifier:
    logger.info("Starting model training")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.debug(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = DecisionTreeClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model
