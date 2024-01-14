from nltk import FreqDist
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
import enchant
import re

# Initialization
data_path = 'DataSets/SpellingCorrection/DataSet/Dataset.data'
misspelled_path = 'DataSets/SpellingCorrection/Dictionary/Text_with_Misspelling.data'
lan = enchant.Dict("en_US")
# lan.check('Hello')
# lan.suggest('Hello')


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


# Example usage
s1 = "kitten"
s2 = "sitting"

distance = damerau_levenshtein_distance(s1, s2)
print(f"Damerau–Levenshtein distance between '{s1}' and '{s2}': {distance}")

# Unigram Probability
with open(data_path, 'r', encoding='utf-8') as file:
    data = file.read()

words = word_tokenize(data)

freq_dist = FreqDist(words)

# print("Unigram frequencies:")
# for word, frequency in freq_dist.items():
#     print(f"{word}: {frequency}")


# Misspelling word
with open(misspelled_path, 'r', encoding='utf-8') as file:
    misspelled_data = file.read()

pattern = re.compile(r'<ERR targ=([^>]+)>([^<]+)</ERR>')
matches = pattern.findall(misspelled_data)

for targ, word in matches:
    print(f"Targ: {targ}, Word: {word}")
