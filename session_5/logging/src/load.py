import logging
import pandas as pd
from metadata import DATASETS_FOLDER

logger = logging.getLogger(__name__)


def load_data(file_name: str) -> pd.DataFrame:
    logger.info(f"Loading dataset: {file_name}")
    df = pd.read_csv(f"{DATASETS_FOLDER}/{file_name}")
    logger.info(f"Dataset loaded with shape {df.shape}")
    return df
