from collections import defaultdict
import string


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words


def create_word_dictionary(text_content):
    words = preprocess_text(text_content)
    word_dictionary = defaultdict(int)
    for word in words:
        word_dictionary[word] += 1
    return word_dictionary


file_path = "DataSets/Classification-Train And Test/Comp.graphics/data554.txt"

# Read File
with open(file_path, 'r', encoding='utf-8') as file:
    test_set_text = file.read()

# Create Dictionary from TestSet
test_set_dictionary = create_word_dictionary(test_set_text)

# نمایش دیکشنری
print("Test set dictionary:")
for word, count in test_set_dictionary.items():
    print(f"{word}: {count}")