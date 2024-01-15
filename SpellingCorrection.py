from collections import Counter
# nltk.download('punkt')
import enchant
import re

# Initialization
data_path = 'DataSets/SpellingCorrection/DataSet/Dataset.data'
misspelled_path = 'DataSets/SpellingCorrection/Dictionary/Text_with_Misspelling.data'
dictionary_path = 'DataSets/SpellingCorrection/Dictionary/dictionary.data'
lan = enchant.Dict("en_US")
# lan.check('Hello')
# lan.suggest('Hello')


# Damerau-Levenshtein Edit distance
def damerau_levenshtein_distance(s1, s2):
    """
    Calculate the Damerau–Levenshtein distance between two strings.
    """
    len_s1 = len(s1)
    len_s2 = len(s2)
    d = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    for i in range(len_s1 + 1):
        d[i][0] = i
    for j in range(len_s2 + 1):
        d[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # transposition

    return d[len_s1][len_s2]


# Unigram Probability
with open(data_path, 'r', encoding='utf-8') as file:
    data = file.read()

words = data.split(' ')
unigram = Counter(words)
total_length = len(words)


# Calculate unigram probabilities
def modify_values(value):
    return value / total_length


for key, value in unigram.items():
    unigram[key] = modify_values(value)


# Misspelling word
with open(misspelled_path, 'r', encoding='utf-8') as file:
    misspelled_data = file.read()

pattern = re.compile(r'<ERR targ=([^>]+)>([^<]+)</ERR>')
matches = pattern.findall(misspelled_data)


# Check Method
def check_edit(str1, str2):     # str1 = Error, str2 = Candidate Correction
    if len(str1) > len(str2):
        return 'ins'
    elif len(str1) < len(str2):
        return 'del'
    elif len(str1) == len(str2):
        word_count = 0
        for char1, char2 in zip(str1, str2):
            if char1 != char2:
                word_count += 1
        if word_count == 1:
            return 'sub'
        elif word_count == 2:
            return 'trans'
        else:
            pass
    else:
        pass


# print Errors
# for targ, word in matches:
#     print(f'Error: {word}')


def clean_word(word):
    # Remove leading and trailing whitespaces
    return word.strip()


for i in range(len(matches)//10):
    s1 = clean_word(matches[i][1])
    s2 = clean_word(matches[i][0])
    Candidate_list = lan.suggest(s1)
    print(matches[i])
    for j in range(len(Candidate_list)):
        Candidate_list[j] = Candidate_list[j].lower()
        distance = damerau_levenshtein_distance(s1, Candidate_list[j])
        edit_operation = check_edit(s1, Candidate_list[j])
        if distance == 1:
            print(f"Levenshtein distance between '{s1}' and '{Candidate_list[j]}': {distance} and op: {edit_operation}")
