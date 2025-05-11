# TODO: move in a "make client dependencies function"m maybe under utils
import os
import urllib


def download_if_not_exists(file_path: str, file_url: str):
    """
    Checks if a file exists at the given path. If it does not, downloads it from the specified URL.

    Parameters:
    - file_path (str): The local path to check and save the file.
    - file_url (str): The URL from which to download the file.
    """
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Downloading from {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download file: {e}")
    else:
        print(f"'{file_path}' already exists.")

def build_fl_dependencies():
    download_if_not_exists(
        "vision_transformer.py",
        "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/vision_transformer.py"
    )

    download_if_not_exists(
        "utils.py",
        "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/utils.py"
    )