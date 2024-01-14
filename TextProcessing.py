from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
import string
import os

'''
    For Normalize Data We need something like:
        1. Convert to lowercase
        2. Remove digits
        3. Remove punctuation
        4. Tokenize Data
        5. Stemming Data ( With Porter Algorithm )
'''


class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


def print_colored(text, color):
    print(color + text + Colors.RESET)


def display_menu():
    print_colored("Welcome to NLP Project", Colors.RED)
    print_colored("Text Processing and Normalize Data\n", Colors.BLUE)
    print_colored("1. Initial Data\n2. Lower-Case\n3. No-Digits\n4. No-Punctuation\n"
                  "5. Tokenize-Data\n6. Token-Count\n7. Stemming-Tokens\n"
                  "8. Tokens VS PorterStemmer", Colors.GREEN)


def count_tokens(tokens):
    # Count the occurrences of each token
    token_counts = Counter(tokens)
    return token_counts


# Read DataSet
with open('DataSets/TextProcessing/TextProcessing.txt', 'r', encoding='utf-8') as file:
    input_str = file.read()

# Convert to lowercase
result_lower = input_str.lower()

# Remove digits
result_no_digits = re.sub(r'\d+', '', result_lower)

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
result_no_punctuation = result_no_digits.translate(translator)

# Tokenize Data
tokens = re.findall(r'\b\w+\b', result_no_punctuation)

token_counts = count_tokens(tokens)

sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

stemmer = PorterStemmer()

singles = [stemmer.stem(token) for token in tokens]

# print(singles)


while True:
    display_menu()
    user_choice = input("Choose an option (or 'q' to quit): ")

    if user_choice == '1':
        print(input_str)
    elif user_choice == '2':
        print(result_lower)
    elif user_choice == '3':
        print(result_no_digits)
    elif user_choice == '4':
        print(result_no_punctuation)
    elif user_choice == '5':
        print(tokens)
    elif user_choice == '6':
        for token, count in sorted_tokens:
            print(f"{token}: {count} times")
    elif user_choice == '7':
        print(singles)
    elif user_choice == '8':
        print(tokens)
        print(singles)
    elif user_choice.lower() == 'q':
        print_colored("Exiting the program. Goodbye!", Colors.RED)
        break

    # Input for Clear Terminal
    print_colored("Press Enter to continue...", Colors.RED)
    input()
    # os.system('cls')
    os.system('cls')
