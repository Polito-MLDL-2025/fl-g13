import os
import random

# Sample lists of funny adjectives and nouns
adjectives = [
    "sleepy", "fluffy", "soggy", "funky", "silly", "breezy", "happy", "jumpy",
    "zesty", "quirky", "snazzy", "witty", "spunky", "cheeky", "perky", "groovy",
    "zany", "bubbly", "chirpy", "dizzy", "giddy", "peppy", "sassy", "wacky", "zippy", "jolly",
    "bouncy", "cranky", "dorky", "frosty", "grumpy", "itchy", "jazzy", "kooky", "loopy",
    "mushy", "nutty", "plucky", "rowdy", "sneezy", "spooky"
]
nouns = [
    "panda", "lizard", "banana", "rocket", "octopus", "cookie", "wizard", "turtle", "pizza",
    "unicorn", "sloth", "muffin", "giraffe", "penguin", "dolphin", "cactus", "marshmallow",
    "axolotl", "hamster", "platypus", "narwhal", "hedgehog", "koala", "otter", "meerkat",
    "alpaca", "chinchilla", "wombat", "toucan", "lemur", "manatee", "aardvark", "quokka"
]

def generate_goofy_name(folder_path=None):
    taken_names = set()
    
    if not folder_path: return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder path '{folder_path}' does not exist or is not a directory.")
    try:
        taken_names = {f.split('_epoch_')[0] for f in os.listdir(folder_path) if f.endswith('.pth')}
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while listing files in '{folder_path}': {e}")
    
    max_attempts = 1000
    for _ in range(max_attempts):
        name = f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(10, 99)}"
        if name not in taken_names:
            return name
    raise RuntimeError(f"Failed to generate a unique name after {max_attempts} attempts. All names could be already taken.")
