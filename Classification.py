from collections import defaultdict
import glob
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


def get_total_word_dictionary(folder_path):
    total_word_dictionary = defaultdict(int)
    file_paths = glob.glob(f"{folder_path}/*.txt")
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            file_dictionary = create_word_dictionary(file_content)
            for word, count in file_dictionary.items():
                total_word_dictionary[word] += count

    return total_word_dictionary


def get_total_word_count(folder_path):
    total_word_count = 0
    file_paths = glob.glob(f"{folder_path}/*.txt")
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            words = preprocess_text(file_content)
            total_word_count += len(words)

    return total_word_count


Comp_folder_path = "DataSets/Classification-Train And Test/Comp.graphics"
Comp_Dict = get_total_word_dictionary(Comp_folder_path)
Comp_Count = get_total_word_count(Comp_folder_path)
Comp_V = len(Comp_Dict)

# Laplace Smoothing
for word in Comp_Dict:
    Comp_Dict[word] += 1
    Comp_Dict[word] /= (Comp_Count + Comp_V)

print("Test set dictionary:")
for word, count in Comp_Dict.items():
    print(f"{word}: {count}")
