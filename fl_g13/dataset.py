import os
from pathlib import Path

from torchvision import datasets, transforms

from loguru import logger
from tqdm import tqdm
import typer

from fl_g13.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset cifar100...")
    train_dataset = download_cifar100(data_dir=RAW_DATA_DIR,train=True)
    test_dataset = download_cifar100(data_dir=RAW_DATA_DIR,train=False)
    logger.success("Downloading dataset complete.")
    # logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Processing dataset complete.")
    # -----------------------------------------

def download_cifar100(data_dir="data/raw", train=True):
    """
    Downloads the CIFAR-100 dataset into the specified directory.
    
    Args:
        data_dir (str): Directory to store the downloaded data.
        train (bool): If True, download the training set; else the test set.
    
    Returns:
        dataset (torchvision.datasets.CIFAR100): The downloaded dataset.
    """
    # Define transformation (you can adjust this for your training needs)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    print(f"{'Training' if train else 'Test'} set downloaded to: {data_dir}")
    return dataset

if __name__ == "__main__":
    app()
