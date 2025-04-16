import random
import string

# Sample lists of funny adjectives and nouns
adjectives = ["sleepy", "fluffy", "soggy", "funky", "silly", "breezy", "happy", "jumpy"]
nouns = ["panda", "lizard", "banana", "rocket", "octopus", "cookie", "wizard", "turtle", "pizza"]

def generate_gooofy_name():
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    number = ''.join(random.choices(string.digits, k=2))  # Generates a 3-digit number
    return f"{adjective}_{noun}_{number}"