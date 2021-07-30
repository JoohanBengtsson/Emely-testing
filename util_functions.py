# Util functions are help functions which can be reached and used from the entire project.
# Functions which are assumed to be usable at several times during the project are to be placed here.
import math
import random
# For question answering
import sys
import time
from os import path

import requests
import torch.cuda
import config

sys.path.append(path.abspath("BERT-SQuAD"))
from bert import QA

# Check similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# -------------------------- Auxiliary variables -----------------------

debug_mode = False


# -------------------------- Util functions -----------------------


# Method that extracts the question from any string_array, containing one or multiple strings. Precondition: questions
# end with a '?'
def extract_question(string_array):
    extracted_questions = []

    # If string_array is truly an array, end is set to the length. If it is not an array, it is inserted into an array.
    if isinstance(string_array, list):
        end = len(string_array)
    else:
        end = 1
        string_array = [string_array]

    # Looping over the sentences in the conversation array
    for index in range(end):

        # Lower-case to disregard from that.
        sentence_obj = string_array[index].lower()

        # Counts how many questions there are in a sentence
        count_question_mark = sentence_obj.count('?')

        # Splits the sentence using '?' and leaves out the part after the last '?'
        split_array = sentence_obj.split('?')
        split_array = split_array[0:count_question_mark]
        # print("split_array: " + str(split_array))

        # The split array after splitting using '?' is looped over, to cleanse it from ', . !' to isolate the question
        for sent in split_array:
            # Every sentence is split using ', . !' and then the part after the punctuation mark is kept - the question
            temp_sent = sent.split('.')
            temp_sent = temp_sent[len(temp_sent) - 1]
            temp_sent = temp_sent.split(',')
            temp_sent = temp_sent[len(temp_sent) - 1]
            temp_sent = temp_sent.split('!')
            temp_sent = temp_sent[len(temp_sent) - 1]
            if temp_sent[0] == ' ':
                temp_sent = temp_sent[1:len(temp_sent)]
            extracted_questions.append(temp_sent)
    return extracted_questions


# Checks the length of any string array and returns the end of an array containing less or equal to the maximum
# max_length amount of tokens.
def check_length_str_array(conv_array, max_length):
    sum_tokens = 0
    for index in range(len(conv_array) - 1, -1, -1):
        sentence = conv_array[index]
        len_sentence_tokens = len(sentence)
        if (sum_tokens + len_sentence_tokens) <= max_length:
            sum_tokens = sum_tokens + len_sentence_tokens
        else:
            return conv_array[(index + 1):len(conv_array)]
    return conv_array


# Method for interpreting the coherence-points achieved using BertForNextSentencePrediction.
def judge_coherences(nsp_points, chatter_index):
    # Since Chatter1 initiates the conversation, the first answer is a conv-starter and thus not assessed.
    if chatter_index == 1:
        coherence_array = ['-']
    else:
        coherence_array = []

    # In order to present the coherence, the result of BERT NSP is classified using 5 labels, namely:
    # ['Most likely a coherent response', 'Likely a coherent response', 'Uncertain result', 'Unlikely a coherent
    # response', 'Most unlikely a coherent response']
    for nsp in nsp_points:
        if nsp > 6:
            coherence_array.append('Most likely coherent')
        elif nsp > 1:
            coherence_array.append('Likely coherent')
        elif nsp > -1:
            coherence_array.append('Uncertain result')
        elif nsp > -6:
            coherence_array.append('Unlikely coherent')
        else:
            coherence_array.append('Most unlikely coherent')
    return coherence_array


# Method for converting the conversation array to a string.
def array2string(conv_array):
    # Converts the conversation array to a string separated by newline
    conv_string = ' '.join([str(elem) + '\n' for elem in conv_array])
    conv_string = conv_string[:len(conv_string) - 1]
    return conv_string


# Checks the similarity between two lists of sentences. Each element is compared to the element of
# the other list with the same index.
def check_similarity(sentences1, sentences2):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    return [float(cosine_scores[i][i]) for i in range(len(cosine_scores))]


# Extract if the answer is yes or no. Very simple, but effective so far
def binaryQA(answers):
    results = []  # Array of binary answers
    answers = [a.split(".")[0] for a in answers]  # Extract first sentence
    negatives = ["no", "do not", "don ' t"]  # Bag of negative words
    for answer in answers:
        if any(word in answer for word in negatives):
            results.append(False)
        else:
            results.append(True)
    return results


# Answer a question about a certain answer string
def openQA(answers, question):
    if debug_mode:
        start_time = time.time()
    modelQA = QA('BERT-SQuAD/model')

    if debug_mode:
        qa_time = time.time() - start_time
        print('qa_time: ' + str(qa_time))
        start_time = time.time()

    for i in range(len(answers)):
        answers[i] = modelQA.predict(answers[i], question)["answer"]

    if debug_mode:
        pred_time = time.time() - start_time
        print('pred_time: ' + str(pred_time))
    return answers


# Return a random element from list including its index
def rand_choice(l):
    idx = random.randint(0, len(l) - 1)
    re = l[idx]
    return re, idx


# Counter for keeping track of which lines that have been used, so that a mix of lines will be used.
counter = {}


# Initiates the counter so that the available lines will be inserted and being started keeping track of.
def init_counter(test_set, sought_info):
    global counter
    temp_list = test_set[sought_info]
    counter[test_set['test']] = {}
    for elem in temp_list:
        counter[test_set['test']][elem] = 0


# Function for finding a sentence to feed the test script which. The chosen sentence will most often be the one that has
# appeared the fewest times.
def selector(test_set):
    global counter
    sentence_list = list(counter[test_set['test']].keys())
    count_list = list(counter[test_set['test']].values())
    min_count = 10000
    min_indices = []
    for i in range(len(count_list)):
        elem = count_list[i]
        if elem == min_count:
            min_indices.append(i)
        elif elem < min_count:
            min_count = elem
            min_indices.clear()
            min_indices.append(i)
    chosen_index = random.choice(min_indices)
    counter[test_set['test']][sentence_list[chosen_index]] = counter[test_set['test']][sentence_list[chosen_index]] + 1
    return sentence_list[chosen_index]


def threshold(results, directed, thresh=0.30):
    bin_results = []
    if directed:
        for result in results:
            if not result:
                bin_results.append("Fail")
            else:
                bin_results.append("Pass")
    else:
        for result in results:
            if result < thresh:
                bin_results.append('Fail')
            else:
                bin_results.append('Pass')
    return bin_results


def create_column(results, idx, n):
    # Makes a list of results and indices into a column
    column = [None] * n
    for i in range(len(idx)):
        column[idx[i]] = results[i]
    return column


def extract_answers(conv_chatter, test_ids, test_set_id):
    # Extract the answers
    test_idx = []
    for i in range(len(test_ids)):
        if test_ids[i] == test_set_id:
            test_idx.append(i)
    answers = [conv_chatter[idx] for idx in test_idx]
    return answers, test_idx


# The method takes any sentence and inserts typing mistakes.
# sentence                      the sentence that should be made wrong.
# percentage_mistyped_words     is the share of the amount of words in which typing mistakes shall be inserted.
# amount_inserted_typos         how many typos that should be inserted per word.
# returns: the transformed sentence
def insert_typing_mistake(sentence, percentage_mistyped_words=None, amount_inserted_typos=None):
    # Splits sentence into an array
    sentence_array = sentence.split()

    # If parameters are not externally decided, they will be randomized here.
    if percentage_mistyped_words is None:
        percentage_mistyped_words = random.random()

    if amount_inserted_typos is None:
        amount_inserted_typos = math.ceil(5 * random.random())

    # Array for storing the values that were used
    values_used = [percentage_mistyped_words, amount_inserted_typos]

    # Calculates how many words that should be typo-fied
    amount_mistyped_words = round(percentage_mistyped_words * len(sentence_array))

    # Produces a list of indices which then is shuffled randomly, so that words can be chosen from a permutation list
    indices_list = list(range(0, len(sentence_array)))
    random.shuffle(indices_list)

    # Loops over the amount of words that should be mistyped. Chooses a word randomly
    for i in range(amount_mistyped_words):
        chosen_word_index = indices_list[i]
        chosen_word = sentence_array[chosen_word_index]

        # Loops over the amount of typos to be inserted per word. Chooses randomly between three types of typos
        for j in range(amount_inserted_typos):
            typo_prob = random.random()
            typo_index = math.floor(len(chosen_word) * random.random())

            # Randomizes what token to insert or swap for, if it should be inserted or swapped for.
            randomized_token_index = ord('a') + math.floor((ord('z') + 1 - ord('a')) * random.random())

            if typo_prob < 1 / 3:  # Inserts false token
                chosen_word = chosen_word[:typo_index] + chr(randomized_token_index) + chosen_word[typo_index:
                                                                                                   len(chosen_word)]
            elif typo_prob < 2 / 3:  # Removes token
                chosen_word = chosen_word[:typo_index] + chosen_word[typo_index + 1:len(chosen_word)]

            else:  # Switches tokens
                chosen_word = chosen_word[:typo_index] + chr(randomized_token_index) + chosen_word[typo_index + 1:
                                                                                                   len(chosen_word)]
            sentence_array[chosen_word_index] = chosen_word
    return ''.join(elem + ' ' for elem in sentence_array), values_used


# Method for introducing word order swaps in any sentence, inserted randomly.
# sentence              the sentence in which a word order swap should be introduced in.
def insert_word_order_swap(sentence, amount_swaps=None):
    sentence_array = sentence.split()

    if amount_swaps is None:
        amount_swaps = round((len(sentence_array) - 1) * random.random())

    for i in range(amount_swaps):
        swap_index = math.floor((len(sentence_array) - 1) * random.random())
        temp_word = sentence_array[swap_index]
        sentence_array[swap_index] = sentence_array[swap_index + 1]
        sentence_array[swap_index + 1] = temp_word
    return ''.join(elem + ' ' for elem in sentence_array), amount_swaps


# Method for inserting a blank space, masking a word. Used for checking how the chatbot deals with such sentences.
def insert_masked_words(sentence, amount_masked=None):
    sentence_array = sentence.split()

    if amount_masked is None:
        amount_masked = round(len(sentence_array) * random.random())

    for i in range(amount_masked):
        mask_index = math.floor(len(sentence_array) * random.random())
        sentence_array[mask_index] = " "
    return ''.join(elem + ' ' for elem in sentence_array), amount_masked


def find_synonym(word):
    url = "https://api.datamuse.com/words"
    json_obj = {
        "ml": word
    }
    r = requests.get(url, json_obj)
    response = r.json()['text']
    return response


counter_values_used = {}


def log_values_used(test_case, index, values_used):
    if test_case not in counter_values_used.keys():
        counter_values_used[test_case] = {}
    counter_values_used[test_case][index] = values_used


def present_values_used(data_frame, test_ids, test_case):
    values_column = []
    set_values_used = counter_values_used[test_case]
    for i in range(len(test_ids)):
        if test_ids[i] % 1 != 0:
            temp = set_values_used.get(i)
            temp_string = ''.join(str(round(elem, 2)) + ':' for elem in temp)
            values_column.append(temp_string[0:len(temp_string) - 1])
        else:
            values_column.append('-')
    data_frame.insert(1, 'Values used for ' + test_case, values_column)
    return data_frame


def ux_test_analysis(data_frame, conv_chatter, test_ids, test_sets, test_case):
    for test_set in test_sets:
        # Extract the answers only given after the question
        test_number = test_case[3]
        answers, test_idx = extract_answers(conv_chatter, test_ids,
                                            2000000 + int(test_number) * 10000 + test_set["id"] + 0.5)

        if not len(answers) > 0:
            continue

        if not test_set["directed"]:
            # Reduce the answer to the specific answer to the question.
            interpret = openQA(answers, test_set["QA"])

            results = check_similarity([test_set["answer"]] * len(interpret), interpret)

            if config.show_binary:
                bin_results = threshold(results, False, thresh=0.3)

        else:
            # Check whether the answer is true
            interpret = binaryQA(answers)

            results = [i == test_set["answer"] for i in interpret]

            if config.show_binary:
                bin_results = threshold(results, True)

        # Add the results to the data frame. Rows outside of the test gets the value None
        if config.show_interpret:
            interpret = create_column(interpret, test_idx, len(conv_chatter))

            data_frame.insert(1, "MLU" + test_number + "TC1 - " + str(test_set["id"]), interpret)

        if config.show_detailed:
            results = create_column(results, test_idx, len(conv_chatter))

            data_frame.insert(1, "MLU" + test_number + "TC1 some typo's - " + str(test_set["id"]), results)

        if config.show_binary:
            bin_results = create_column(bin_results, test_idx, len(conv_chatter))

            data_frame.insert(1, "MLU" + test_number + "TC1 - " + str(test_set["id"]), bin_results)

    data_frame = present_values_used(data_frame, test_ids, 'MLU' + test_number + 'TC1')
    return data_frame
