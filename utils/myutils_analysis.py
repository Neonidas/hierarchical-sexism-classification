from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import re


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def cleaning_data(row, stopwords, WNL):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(row))

    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if word in stopwords:
            continue
        elif tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(WNL.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    return lemmatized_sentence


def get_data_cleaned(text):
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)

    stopwords = nltk.corpus.stopwords.words('english')

    WNL = WordNetLemmatizer()

    data_cleaned = text.apply(lambda x: cleaning_data(x, stopwords, WNL))
    return data_cleaned


def find_confusion_matrix_lists(true_labels, predicted_labels, data, category=1):
    """
    Takes a list of true labels, a list of predicted_labels, the data and optionally a category and
    puts the data into its correct new list, that is either true positive(TP), true negative(TN),
    false positive(FP) or false negative(FN). It returns these 4 lists
    """
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == category:
            if predicted_labels[i] == true_labels[i]:
                true_positives.append(data[i])
            else:
                false_positives.append(data[i])
        elif true_labels[i] == category:
            false_negatives.append(data[i])
        else:
            true_negatives.append(data[i])
    return true_positives, true_negatives, false_positives, false_negatives

def set_to_one(labels):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 0:
            new_labels.append(0)
        else:
            new_labels.append(1)
    return new_labels

def set_to_category(labels):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 1 or labels[i] == 2:
            new_labels.append(1)
        elif labels[i] == 3 or labels[i] == 4 or labels[i] == 5:
            new_labels.append(2)
        elif labels[i] == 6 or labels[i] == 7 or labels[i] == 8 or labels[i] == 9:
            new_labels.append(3)
        elif labels[i] == 10 or labels[i] == 11:
            new_labels.append(4)
        elif labels[i] == 0:
            new_labels.append(0)
    return new_labels

def get_category_name(labels):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 1:
            new_labels.append("1.1")
        elif labels[i] == 2:
            new_labels.append("1.2")
        elif labels[i] == 3:
            new_labels.append("2.1")
        elif labels[i] == 4:
            new_labels.append("2.2")
        elif labels[i] == 5:
            new_labels.append("2.3")
        elif labels[i] == 6:
            new_labels.append("3.1")
        elif labels[i] == 7:
            new_labels.append("3.2")
        elif labels[i] == 8:
            new_labels.append("3.3")
        elif labels[i] == 9:
            new_labels.append("3.4")
        elif labels[i] == 10:
            new_labels.append("4.1")
        elif labels[i] == 11:
            new_labels.append("4.2")
        elif labels[i] == 0:
            new_labels.append("0")
    return new_labels

def find_all_predicted_in_category(predicted_labels, data, category):
    predicted_in_category = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == category:
            predicted_in_category.append(data[i])
    return predicted_in_category


def create_word_cloud(text, max_words=200):
    """
    Creates a word cloud given a text and optionally the maximum amount of words to be shown in
    the word cloud
    """
    all_words = ' '.join(map(str, text))

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          min_font_size=10).generate(all_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def common_words(text, no_words):
    """
    common_words counts how often specific words occur in the text and returns a sorted list from
    the largest to the lowest amount of occurrences.
    """
    split_it = '\n'.join(text).split()
    counter = Counter(split_it)
    most_occur = counter.most_common(no_words)
    print(f'In total there is {len(split_it)} words in the array')
    print(most_occur)
    return most_occur


def common_bigrams(text, no_words):
    """
    bigram_counts counts how often a specific combination of two words occur in the text and
    returns a sorted list from the largest to the lowest amount of occurrences.
    """
    bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    print(f'In total there is {len(bigrams)} bigrams in the array')
    print(Counter(bigrams).most_common(no_words))


def show_word_clouds_confusion_matrix(true_labels, predicted_labels, test_data, category=1, max_words=200):
    """
    Shows the four word clouds which are created by create_word_cloud.
    """
    true_positive, true_negative, false_positive, false_negative = \
        find_confusion_matrix_lists(true_labels, predicted_labels, test_data, category)
    print("True Positive")
    if not true_positive:
        print("No text was said to be true positive")
    else:
        create_word_cloud(true_positive, max_words)
    print()
    print("True Negative")
    if not true_negative:
        print("No text was said to be true negative")
    else:
        create_word_cloud(true_negative, max_words)
    print()
    print("False Positive")
    if not false_positive:
        print("No text was said to be false positive")
    else:
        create_word_cloud(false_positive, max_words)
    print()
    print("False Negative")
    if not false_negative:
        print("No text was said to be false negative")
    else:
        create_word_cloud(false_negative, max_words)
    print()


def show_word_clouds_categories(true_labels, predicted_labels, test_data, max_words=200):
    categories = true_labels.unique()
    for c in range(len(categories)):
        cat = find_all_predicted_in_category(predicted_labels, test_data, c)
        if not cat:
            print("No text in category: " + str(c))
        else:
            print("category: " + str(c))
            create_word_cloud(cat,max_words)

def show_word_clouds_categories_true(true_labels, test_data, max_words=200):
    categories = true_labels.unique()
    for c in range(len(categories)):
        cat = find_all_predicted_in_category(true_labels, test_data, c)
        if not cat:
            print("No text in category: " + str(c))
        else:
            print("category: " + str(c))
            create_word_cloud(cat,max_words)

def show_common_words_confusion_matrix(true_labels, predicted_labels, test_data, no_words, category=1):
    """
    show_common_words prints the result of common_words within their given category, that is
    TP, TN, FP and FN
    """
    true_positive, true_negative, false_positive, false_negative = \
        find_confusion_matrix_lists(true_labels, predicted_labels, test_data, category)
    print("True Positive")
    if not true_positive:
        print("No text was said to be true positive")
    else:
        common_words(true_positive, no_words)
    print()
    print("True Negative")
    if not true_negative:
        print("No text was said to be true negative")
    else:
        common_words(true_negative, no_words)
    print()
    print("False Positive")
    if not false_positive:
        print("No text was said to be false positive")
    else:
        common_words(false_positive, no_words)
    print()
    print("False Negative")
    if not false_negative:
        print("No text was said to be false negative")
    else:
        common_words(false_negative, no_words)


def show_common_words_categories(true_labels, predicted_labels, test_data, no_words=20):
    categories = true_labels.unique()
    for c in range(len(categories)):
        cat = find_all_predicted_in_category(predicted_labels, test_data, c)
        if not cat:
            print("No text in category: " + str(c))
        else:
            print("category: " + str(c))
            common_words(cat,no_words)

def show_common_words_categories_true(true_labels, test_data, no_words=20):
    categories = true_labels.unique()
    for c in range(len(categories)):
        cat = find_all_predicted_in_category(true_labels, test_data, c)
        if not cat:
            print("No text in category: " + str(c))
        else:
            print("category: " + str(c))
            common_words(cat,no_words)


def show_common_bigrams_confusion_matrix(true_labels, predicted_labels, test_data, no_words, category=1):
    """
    show_common_bigrams prints the result of bigram_counts within their given category, that is
    TP, TN, FP and FN
    """
    true_positive, true_negative, false_positive, false_negative = \
        find_confusion_matrix_lists(true_labels, predicted_labels, test_data, category)
    print("True Positive")
    if not true_positive:
        print("No text was said to be true positive")
    else:
        common_bigrams(true_positive, no_words)
    print()
    print("True Negative")
    if not true_negative:
        print("No text was said to be true negative")
    else:
        common_bigrams(true_negative, no_words)
    print()
    print("False Positive")
    if not false_positive:
        print("No text was said to be false positive")
    else:
        common_bigrams(false_positive, no_words)
    print()
    print("False Negative")
    if not false_negative:
        print("No text was said to be false negative")
    else:
        common_bigrams(false_negative, no_words)


def show_common_bigrams_categories(true_labels, predicted_labels, test_data, no_words=20):
    categories = true_labels.unique()
    for c in range(len(categories)):
        cat = find_all_predicted_in_category(predicted_labels, test_data, c)
        if not cat:
            print("No text in category: " + str(c))
        else:
            print("category: " + str(c))
            common_bigrams(cat, no_words)


def calculate_wrongly_classified(true_labels, predicted_labels, text, correct_category, wrong_category):
    array = []
    for i in range(len(predicted_labels)):
        if true_labels[i] == correct_category and predicted_labels[i] == wrong_category:
            array.append(text[i])
    return len(array)


def get_wrongly_classified(true_labels, predicted_labels, correct_category, text):
    categories = true_labels.unique()
    wrongly_classified = {}
    for j in range(1, len(categories)):
        wrongly_classified[j] = calculate_wrongly_classified(true_labels, predicted_labels, text, correct_category, j)
    return wrongly_classified
