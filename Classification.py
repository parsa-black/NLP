import glob
import string
from collections import defaultdict
import math


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
        # Multiply by Laplace smoothing term in the denominator
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

Count_V = len(Talk_Dict) + len(Soc_Dict) + len(Sci_Dict) + len(Rec_Dict) + len(Comp_Dict)

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
    "Comp": Count_V,
    "Rec": Count_V,
    "Sci": Count_V,
    "Soc": Count_V,
    "Talk": Count_V
}

class_doc = {
    "Comp": 8 / 94,
    "Rec": 15 / 94,
    "Sci": 14 / 94,
    "Soc": 27 / 94,
    "Talk": 30 / 94
}


# Test-set
def classify_test_set(test_set_path, classes_dicts, classes_count, classes_v, class_docs):
    with open(test_set_path, 'r', encoding='utf-8') as file:
        test_content = file.read()
        test_words = preprocess_text(test_content)

    total_log_probability = defaultdict(float)

    for class_name, class_dict in classes_dicts.items():
        class_log_probability = math.log(class_docs[class_name])
        for word in test_words:
            if word in class_dict:
                # Sum the logarithms of probabilities
                class_log_probability += math.log(class_dict[word])
            else:
                # Add Laplace smoothing term for unseen words
                class_log_probability += math.log(1 / (classes_count[class_name] + classes_v[class_name]))

        total_log_probability[class_name] = class_log_probability

    return total_log_probability


# Test-set paths
ts480_path = 'DataSets/Classification-Train And Test/Comp.graphics/test/data480.txt'
ts488_path = 'DataSets/Classification-Train And Test/Comp.graphics/test/data488.txt'
ts3982_path = 'DataSets/Classification-Train And Test/rec.autos/test/data3982.txt'
ts3992_path = 'DataSets/Classification-Train And Test/rec.autos/test/data3992.txt'
ts4011_path = 'DataSets/Classification-Train And Test/rec.autos/test/data4011.txt'
ts6990_path = 'DataSets/Classification-Train And Test/sci.electronics/test/data6990.txt'
ts6993_path = 'DataSets/Classification-Train And Test/sci.electronics/test/data6993.txt'
ts7016_path = 'DataSets/Classification-Train And Test/sci.electronics/test/data7016.txt'
ts8750_path = 'DataSets/Classification-Train And Test/soc.religion.christian/test/data8750.txt'
ts8752_path = 'DataSets/Classification-Train And Test/soc.religion.christian/test/data8752.txt'
ts8762_path = 'DataSets/Classification-Train And Test/soc.religion.christian/test/data8762.txt'
ts8770_path = 'DataSets/Classification-Train And Test/soc.religion.christian/test/data8770.txt'
ts9891_path = 'DataSets/Classification-Train And Test/talk.politics.mideast/test/data9891.txt'
ts9899_path = 'DataSets/Classification-Train And Test/talk.politics.mideast/test/data9899.txt'
ts9905_path = 'DataSets/Classification-Train And Test/talk.politics.mideast/test/data9905.txt'
ts9906_path = 'DataSets/Classification-Train And Test/talk.politics.mideast/test/data9906.txt'
ts9910_path = 'DataSets/Classification-Train And Test/talk.politics.mideast/test/data9910.txt'

# Classify test sets
ts480_prob = classify_test_set(ts480_path, class_dicts, class_counts, class_V, class_doc)
ts488_prob = classify_test_set(ts488_path, class_dicts, class_counts, class_V, class_doc)
ts3982_prob = classify_test_set(ts3982_path, class_dicts, class_counts, class_V, class_doc)
ts3992_prob = classify_test_set(ts3992_path, class_dicts, class_counts, class_V, class_doc)
ts4011_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts6990_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts6993_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts7016_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts8750_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts8752_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts8762_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts8770_prob = classify_test_set(ts4011_path, class_dicts, class_counts, class_V, class_doc)
ts9891_prob = classify_test_set(ts9891_path, class_dicts, class_counts, class_V, class_doc)
ts9899_prob = classify_test_set(ts9899_path, class_dicts, class_counts, class_V, class_doc)
ts9905_prob = classify_test_set(ts9905_path, class_dicts, class_counts, class_V, class_doc)
ts9906_prob = classify_test_set(ts9906_path, class_dicts, class_counts, class_V, class_doc)
ts9910_prob = classify_test_set(ts9910_path, class_dicts, class_counts, class_V, class_doc)


# Function to find the class with the maximum probability
def find_max_probability(probabilities):
    max_class = max(probabilities, key=probabilities.get)
    max_probability = probabilities[max_class]
    return max_class


# for class_name, log_probability in ts8750_prob.items():
#     print(f"{class_name}: {log_probability}")

# Accuracy
Test_set_Name = [ts480_prob, ts488_prob, ts3982_prob, ts3992_prob, ts4011_prob, ts6990_prob, ts6993_prob, ts7016_prob,
                 ts8750_prob, ts8752_prob, ts8762_prob, ts8770_prob, ts9891_prob, ts9899_prob, ts9905_prob,
                 ts9905_prob, ts9910_prob]
Test_set_Class = ['Comp', 'Comp', 'Rec', 'Rec', 'Rec', 'Sci', 'Sci', 'Sci', 'Soc', 'Soc', 'Soc', 'Soc', 'Talk', 'Talk',
                  'Talk', 'Talk', 'Talk']

correct_predictions = 0
total_test_instances = len(Test_set_Name)

for i in range(total_test_instances):
    predicted_label = find_max_probability(Test_set_Name[i])
    actual_label = Test_set_Class[i]

    if predicted_label == actual_label:
        correct_predictions += 1

accuracy = correct_predictions / total_test_instances

# Print the result
print("Classification:")
for i, test_set_prob in enumerate(Test_set_Name):
    test_set_name = f"Test{i + 1:02d}"
    print(f"{test_set_name} : {find_max_probability(test_set_prob)}")
    print('-' * 50)

# Print Accuracy
print(f"\nAccuracy: {accuracy * 100:.2f}%")
