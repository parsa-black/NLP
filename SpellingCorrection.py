from collections import Counter
from nltk import bigrams
from nltk.tokenize import word_tokenize
# nltk.download('punkt')


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


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data


def calculate_bigram_probabilities(sentences):
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    bigram_list = [list(bigrams(sentence)) for sentence in words]
    flat_bigrams = [bigram for sublist in bigram_list for bigram in sublist]

    bigram_counts = Counter(flat_bigrams)
    unigram_counts = Counter([word for sublist in words for word in sublist])

    bigram_probabilities = {}
    for bigram, count in bigram_counts.items():
        preceding_word = bigram[0]
        probability = count / unigram_counts[preceding_word]
        bigram_probabilities[bigram] = probability

    return bigram_probabilities


# Load candidate words and sentences
candidate_words = load_data('DataSets/SpellingCorrection/Dictionary/dictionary.data')
sentences_with_misspelling = load_data('DataSets/SpellingCorrection/Dictionary/Text_with_Misspelling.data')

# Calculate bigram probabilities
bigram_probabilities = calculate_bigram_probabilities(sentences_with_misspelling)

# Print bigram probabilities
for bigram, probability in bigram_probabilities.items():
    print(f"Bigram: {bigram}, Probability: {probability}")
