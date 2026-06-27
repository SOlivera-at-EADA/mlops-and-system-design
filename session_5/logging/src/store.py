import logging
import joblib
from datetime import datetime
from metadata import MODELS_FOLDER

logger = logging.getLogger(__name__)


def store_model(model, model_name: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"{MODELS_FOLDER}/{model_name}-{timestamp}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model stored at: {model_path}")
