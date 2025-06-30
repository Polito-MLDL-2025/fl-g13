import os
import random
import shutil
import glob

from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from torchvision import datasets
from fl_g13.dataset import train_test_split
from torch.utils.data import Subset


# Sample lists of funny adjectives and nouns
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


def generate_goofy_name(folder_path=None):
    """
    Generates a unique and "goofy" name by combining a random adjective, noun, and a two-digit number.
    
    If a folder path is provided, the function ensures that the generated name does not conflict
    with existing names in the folder (specifically, filenames ending with ".pth" and following
    a specific naming convention).

    Args:
        folder_path (str, optional): The path to a folder where existing names are checked. 
            If None, the function generates a name without checking for conflicts.

    Returns:
        str: A unique goofy name in the format "<adjective>_<noun>_<two-digit-number>".

    Raises:
        FileNotFoundError: If the provided folder path does not exist or is not a directory.
        RuntimeError: If an unexpected error occurs while listing files in the folder, or if
            a unique name cannot be generated after 1000 attempts.
    """
    taken_names = set()

    if not folder_path:
        return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder path '{folder_path}' does not exist or is not a directory."
        )
    try:
        taken_names = {
            f.split("_epoch_")[0] for f in os.listdir(folder_path) if f.endswith(".pth")
        }
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while listing files in '{folder_path}': {e}"
        )

    max_attempts = 1000
    for _ in range(max_attempts):
        name = f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"
        if name not in taken_names:
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
    # Preprocessing pipeline
    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    eval_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
    ])

    cifar100_train_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    cifar100_test = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)
    
    test_dataset = cifar100_test

    if do_full_training:
        # In full training, we return train+test datasets
        train_dataset = cifar100_train_full
        return train_dataset, test_dataset
    else:
        # Create a second source, with eval_transformation
        cifar100_val = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=eval_transform)
        
        # Split
        train_dataset, val_dataset_tmp = train_test_split(cifar100_train_full, 0.8, random_state = random_state)
        
        # Use the same indices to create the validation dataset with the correct tranformations
        val_dataset = Subset(cifar100_val, val_dataset_tmp.indices)
        
        return train_dataset, val_dataset, test_dataset