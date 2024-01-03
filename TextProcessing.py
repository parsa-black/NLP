import re
import string

'''
    For Normalize Data We need something like:
        1. Convert to lowercase
        2. Remove digits
        3. Remove punctuation
        4. Remove WhiteSpaces
'''

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

print(result)
