MODELS_FOLDER = "session_5/logging/models"
DATASETS_FOLDER = "session_5/logging/datasets"
MODEL_NAME = "class_model-Santiago"

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
ONE_HOT_ENCODE_COLUMNS = ["Geography", "Gender"]
MODEL_PARAMS = {
    "max_depth": 6,
    "min_samples_split": 9,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": 42,
}
