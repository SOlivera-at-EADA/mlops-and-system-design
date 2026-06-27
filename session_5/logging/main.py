import logging
from src.load import load_data
from src.transform import Transformer
from src.train import train_model
from src.store import store_model
from src.notifier import Notifier
from metadata import MODEL_NAME

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    try:
        df = load_data(file_name="Churn_Modelling_train_test.csv")
        df = Transformer().transform(df)
        model = train_model(df=df, target_column="Exited")
        store_model(model=model, model_name=MODEL_NAME)
    except Exception as e:
        logging.exception(f"An error occurred during the training pipeline. Error: {e}")
        Notifier(process_name="Training Pipeline").print_console_message()


if __name__ == "__main__":
    main()
