import glob
import string
from collections import defaultdict


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


def apply_laplace_smoothing(word_dict, word_count, v):
    for word in word_dict:
        # Add Laplace smoothing term to the numerator
        word_dict[word] += 1
        # Add Laplace smoothing term to the denominator
        word_dict[word] /= (word_count + v)


# Comp Class
Comp_folder_path = "DataSets/Classification-Train And Test/Comp.graphics"
Comp_Dict = get_total_word_dictionary(Comp_folder_path)
Comp_Count = get_total_word_count(Comp_folder_path)
Comp_V = len(Comp_Dict)
# Laplace Smoothing
apply_laplace_smoothing(Comp_Dict, Comp_Count, Comp_V)

# Rec Class
Rec_folder_path = "DataSets/Classification-Train And Test/rec.autos"
Rec_Dict = get_total_word_dictionary(Rec_folder_path)
Rec_Count = get_total_word_count(Rec_folder_path)
Rec_V = len(Rec_Dict)
# Laplace Smoothing
apply_laplace_smoothing(Rec_Dict, Rec_Count, Rec_V)

# Sci Class
Sci_folder_path = "DataSets/Classification-Train And Test/sci.electronics"
Sci_Dict = get_total_word_dictionary(Sci_folder_path)
Sci_Count = get_total_word_count(Sci_folder_path)
Sci_V = len(Sci_Dict)
# Laplace Smoothing
apply_laplace_smoothing(Sci_Dict, Sci_Count, Sci_V)

# Soc Class
Soc_folder_path = "DataSets/Classification-Train And Test/soc.religion.christian"
Soc_Dict = get_total_word_dictionary(Soc_folder_path)
Soc_Count = get_total_word_count(Soc_folder_path)
Soc_V = len(Soc_Dict)
# Laplace Smoothing
apply_laplace_smoothing(Soc_Dict, Soc_Count, Soc_V)

# Talk Class
Talk_folder_path = "DataSets/Classification-Train And Test/talk.politics.mideast"
Talk_Dict = get_total_word_dictionary(Talk_folder_path)
Talk_Count = get_total_word_count(Talk_folder_path)
Talk_V = len(Talk_Dict)
# Laplace Smoothing
apply_laplace_smoothing(Talk_Dict, Talk_Count, Talk_V)


# Test-set
def classify_test_set(test_set_path, classes_dicts, classes_count, classes_v, total_docs):
    with open(test_set_path, 'r', encoding='utf-8') as file:
        test_content = file.read()
        test_words = preprocess_text(test_content)

    total_probability = defaultdict(float)

    for word in test_words:
        for class_name, class_dict in classes_dicts.items():
            if word in class_dict:
                probability = class_dict[word]
            else:
                probability = 1 / (classes_count[class_name] + classes_v[class_name])
            total_probability[class_name] += probability

    # Calculate class | document probabilities
    for class_name, class_count in classes_count.items():
        total_probability[class_name] += class_count / total_docs

    return total_probability


class_dicts = {
    "Comp": Comp_Dict,
    "Rec": Rec_Dict,
    "Sci": Sci_Dict,
    "Soc": Soc_Dict,
    "Talk": Talk_Dict
}

class_counts = {
    "Comp": Comp_Count,
    "Rec": Rec_Count,
    "Sci": Sci_Count,
    "Soc": Soc_Count,
    "Talk": Talk_Count
}

class_V = {
    "Comp": Comp_V,
    "Rec": Rec_V,
    "Sci": Sci_V,
    "Soc": Soc_V,
    "Talk": Talk_V
}

total_docs = {
    "Comp": 8,
    "Rec": 15,
    "Sci": 14,
    "Soc": 27,
    "Talk": 30
}

ts480_path = 'DataSets/Classification-Train And Test/Comp.graphics/test/data480.txt'
ts480_prob = classify_test_set(ts480_path, class_dicts, class_counts, class_V, sum(total_docs.values()))

ts488_path = 'DataSets/Classification-Train And Test/Comp.graphics/test/data488.txt'
ts488_prob = classify_test_set(ts488_path, class_dicts, class_counts, class_V, sum(total_docs.values()))

ts3982_path = 'DataSets/Classification-Train And Test/rec.autos/test/data3982.txt'
ts3982_prob = classify_test_set(ts3982_path, class_dicts, class_counts, class_V, sum(total_docs.values()))

ts3992_path = 'DataSets/Classification-Train And Test/rec.autos/test/data3992.txt'
ts3992_prob = classify_test_set(ts3982_path, class_dicts, class_counts, class_V, sum(total_docs.values()))


# Print the result
print("Test set classification probabilities:")
for class_name, probability in ts480_prob.items():
    print(f"{class_name}: {probability}")
print('-' * 50)
for class_name, probability in ts488_prob.items():
    print(f"{class_name}: {probability}")
print('-' * 50)
for class_name, probability in ts3982_prob.items():
    print(f"{class_name}: {probability}")
print('-' * 50)
for class_name, probability in ts3992_prob.items():
    print(f"{class_name}: {probability}")
