from collections import Counter
import re
import string

'''
    For Normalize Data We need something like:
        1. Convert to lowercase
        2. Remove digits
        3. Remove punctuation
        4. Remove WhiteSpaces
'''


def count_tokens(tokens):
    # Count the occurrences of each token
    token_counts = Counter(tokens)
    return token_counts


# Read DataSet
with open('DataSets/TextProcessing.txt', 'r', encoding='utf-8') as file:
    input_str = file.read()

# Convert to lowercase
result = input_str.lower()

# Remove digits
result = re.sub(r'\d+', '', result)

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
result = result.translate(translator)

# Remove WhiteSpaces
result = result.strip()

# Tokenize Data
tokens = re.findall(r'\b\w+\b', result)

token_counts = count_tokens(tokens)

sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

print(tokens)

for token, count in sorted_tokens:
    print(f"{token}: {count} times")
