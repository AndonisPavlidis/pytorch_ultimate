from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

def download_kaggle_dataset(dataset_name: str, path: str):
    """
    Downloads a Kaggle dataset using the Kaggle API.

    Args:
        dataset_name (str): The name of the dataset to download.
        path (str): The path to save the downloaded dataset.

    Returns:
        None
    """
    api.dataset_download_files(dataset_name, path, unzip=True)