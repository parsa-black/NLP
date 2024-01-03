from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
import string

'''
    For Normalize Data We need something like:
        1. Convert to lowercase
        2. Remove digits
        3. Remove punctuation
        4. Remove WhiteSpaces
        5. Tokenize Data
        6. Stemming Data ( With Porter Algorithm )
'''


def count_tokens(tokens):
    # Count the occurrences of each token
    token_counts = Counter(tokens)
    return token_counts


# Read DataSet
with open('DataSets/TextProcessing.txt', 'r', encoding='utf-8') as file:
    input_str = file.read()

# Convert to lowercase
result_lower = input_str.lower()

# Remove digits
result_no_digit= re.sub(r'\d+', '', result_lower)

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
result_no_punctuation = result_no_digit.translate(translator)

# Remove WhiteSpaces
result_no_whitespace = result_no_punctuation.strip()

# Tokenize Data
tokens = re.findall(r'\b\w+\b', result_no_whitespace)

token_counts = count_tokens(tokens)

sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

for token, count in sorted_tokens:
    print(f"{token}: {count} times")

stemmer = PorterStemmer()

singles = [stemmer.stem(token) for token in tokens]

print(singles)
