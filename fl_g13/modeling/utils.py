import os
import random
import shutil
import glob


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

def backup(path):
    """
    Backups a file to a 'backup' directory located in the same parent directory.
    The backup directory is created if it does not exist.
    The backup file will have the same name as the original file.
    Args:
        path (str): The path to the file to be backed up.
    Raises:
        ValueError: If the provided path is not a file.
    """
    # Check if the provided path is a file
    if os.path.isfile(path):
        file_to_backup = path
    # Check if the provided path is a directory, and find the most recent file
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'), recursive=False)
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise ValueError(f"No files found in directory: {path}")
        file_to_backup = max(files, key=os.path.getmtime)
        print(f"Directory given. Most recent file selected: {file_to_backup}")
    else:
        raise ValueError(f"Invalid path: {path}")

    # Get directory and filename
    dir_name, file_name = os.path.split(path)

    # Create the backup directory inside the same parent directory
    backup_dir = os.path.join(os.path.dirname(dir_name), 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    # Define the destination path
    dest_path = os.path.join(backup_dir, file_name)

    # Copy the file
    shutil.copy2(path, dest_path)
    print(f"Backed up '{path}' to '{dest_path}'")
