import os
import random
import shutil
import glob

from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from torchvision import datasets
from fl_g13.dataset import train_test_split
from torch.utils.data import Subset

# Sample lists of adjectives and nouns
adjectives = [
    "sleepy",
    "fluffy",
    "soggy",
    "funky",
    "silly",
    "breezy",
    "happy",
    "jumpy",
    "zesty",
    "quirky",
    "snazzy",
    "witty",
    "spunky",
    "cheeky",
    "perky",
    "groovy",
    "zany",
    "bubbly",
    "chirpy",
    "dizzy",
    "giddy",
    "peppy",
    "sassy",
    "wacky",
    "zippy",
    "jolly",
    "bouncy",
    "cranky",
    "dorky",
    "frosty",
    "grumpy",
    "itchy",
    "jazzy",
    "loopy",
    "mushy",
    "nutty",
    "plucky",
    "sneezy",
    "spooky",
]
nouns = [
    "bulbasaur",
    "ivysaur",
    "venusaur",
    "charmander",
    "charmeleon",
    "charizard",
    "squirtle",
    "wartortle",
    "blastoise",
    "caterpie",
    "metapod",
    "butterfree",
    "weedle",
    "kakuna",
    "beedrill",
    "pidgey",
    "pidgeotto",
    "pidgeot",
    "rattata",
    "raticate",
    "spearow",
    "fearow",
    "ekans",
    "arbok",
    "pikachu",
    "raichu",
    "sandshrew",
    "sandslash",
    "nidoran",
    "nidorina",
    "nidoqueen",
    "nidorino",
    "nidoking",
    "clefairy",
]

def generate_unique_name(folder_path=None):
    """
    Generates a unique name by combining a random adjective, noun, and a two-digit number.

    If a folder path is provided, the function ensures that the generated name does not conflict
    with existing file prefixes in the folder or its subdirectories.

    Args:
        folder_path (str, optional): The path to a folder where existing names are checked.
            If the folder does not exist, it will be created. Defaults to None.

    Returns:
        str: A unique name in the format "<adjective>_<noun>_<two-digit-number>".

    Raises:
        RuntimeError: If a unique name cannot be generated after 1000 attempts.
    """
    if not folder_path:
        return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    max_attempts = 1000
    for _ in range(max_attempts):
        name = f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"
        # Check if a file with this prefix already exists in any subdirectory
        conflict_found = False
        for _, _, filenames in os.walk(folder_path):
            if any(f.startswith(f"{name}_") for f in filenames):
                conflict_found = True
                break
        if not conflict_found:
            return name
    raise RuntimeError(
        f"Failed to generate a unique name after {max_attempts} attempts. All names could be already taken."
    )

def backup(path, new_filename=None):
    """
    Backups a file to a 'backup' directory located in the same parent directory.
    If a directory is passed, the most recently modified file is selected.
    
    Args:
        path (str): The path to the file or directory to back up.
        new_filename (str, optional): New name for the backup file. If not provided, the original name is used.
        
    Raises:
        ValueError: If the provided path is invalid or contains no files.
    """
    # Resolve the file to back up
    if os.path.isfile(path):
        file_to_backup = path
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'), recursive=False)
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise ValueError(f"No files found in directory: {path}")
        file_to_backup = max(files, key=os.path.getmtime)
        print(f"Directory given. Most recent file selected: {file_to_backup}")
    else:
        raise ValueError(f"Invalid path: {path}")

    # Extract the directory and original filename
    dir_name = os.path.dirname(file_to_backup)
    original_filename = os.path.basename(file_to_backup)

    # Determine final backup filename
    backup_filename = new_filename if new_filename is not None else original_filename

    # Create the backup directory
    backup_dir = os.path.join(os.path.dirname(dir_name), 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    # Define final destination
    dest_path = os.path.join(backup_dir, backup_filename)

    # Perform the copy
    shutil.copy2(file_to_backup, dest_path)
    print(f"Backed up '{file_to_backup}' to '{dest_path}'")

def get_preprocessing_pipeline(data_dir, random_state=42, do_full_training=False):
    """
    Initializes and returns the datasets for the CIFAR-100 classification task,
    with appropriate data augmentation and preprocessing pipelines.

    This function prepares the training, validation, and test datasets.
    - The training pipeline includes data augmentation (random crop, horizontal flip).
    - The evaluation pipeline (for validation and testing) uses a deterministic crop.
    - All datasets are normalized using ImageNet statistics.

    Args:
        data_dir (str): The root directory where the CIFAR-100 dataset is stored or will be downloaded.
        random_state (int, optional): The seed for the random train/validation split. Defaults to 42.
        do_full_training (bool, optional): If True, the function returns the full training dataset and the test set,
            without creating a validation split. This is useful for final model training. Defaults to False.

    Returns:
        tuple: A tuple containing the datasets.
            - If `do_full_training` is False, it returns (train_dataset, val_dataset, test_dataset).
            - If `do_full_training` is True, it returns (train_dataset, test_dataset).
    """
    # Define the preprocessing pipeline for training
    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    # Define the preprocessing pipeline for evaluation (validation and testing)
    eval_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
    ])

    # Load the full training and test sets with their respective transforms
    full_train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)

    if do_full_training:
        # Return the full training set and the test set
        return full_train_set, test_set
    else:
        # To create a validation set with evaluation transforms, we first need to get the indices
        # from a split of the training data. We create a temporary dataset with evaluation transforms
        # to apply the same indices later.
        full_train_set_with_eval_transform = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=eval_transform)

        # Split the training data to get the indices for the training and validation sets
        train_subset, val_subset_with_train_transform = train_test_split(full_train_set, 0.8, random_state=random_state)

        # Create the validation set using the indices from the split, but with the evaluation transforms
        val_subset = Subset(full_train_set_with_eval_transform, val_subset_with_train_transform.indices)

        return train_subset, val_subset, test_set