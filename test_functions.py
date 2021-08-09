# Test functions are the functions used for assessing a conversation.
# They are linked to the requirements presented in the SRS.

import requests
import pandas as pd
import torch
import os
# from transformers import BertTokenizer, BertForNextSentencePrediction  # BlenderbotConfig, pipeline, \
from detoxify import Detoxify
from collections import Counter
from nltk import ngrams

import config
from config import *
import main
import util_functions

# --------------------------- External modules ---------------------------
# These rather slow-loaded models are not loaded if present_metrics is not true, to reduce the startup time when working
# on the code.
# Initiates Bert for Next Sentence Prediction (NSP) and stores the result
# bert_type = 'bert-base-uncased'
# bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
# bert_model = BertForNextSentencePrediction.from_pretrained(bert_type)

# To specify the device the Detoxify-model will be allocated on (defaults to cpu), accepts any torch.device input
if torch.cuda.is_available():
    model = Detoxify('unbiased', device='cuda')
else:
    model = Detoxify('unbiased')


# --------------------------- Test functions ---------------------------


# Analyzes responses of chatter number chatter_index w.r.t the whole conversation that has passed.
def MLI2TC1(conv_array, data_frame):
    # Array for collecting the score
    print("     MLI2TC1")
    # chatter_index = 2
    nsp_points = []

    for index in range(1, len(conv_array), 2):
        relevant_conv_array = util_functions.check_length_str_array(conv_array[0:(index - 1)], 512)

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(
                len(relevant_conv_array) - 1)]])  # conv_array[0:(index - 1)]])
        chatter_response = conv_array[index]

        # Setting up the tokenizer
        # inputs = bert_tokenizer(conv_string_input, chatter_response, return_tensors='pt')

        # Predicting the coherence score using Sentence-BERT
        # outputs = bert_model(**inputs)
        temp_list = util_functions.nsp(conv_string_input, chatter_response)

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    coherence_array = util_functions.judge_coherences(nsp_points, 2)
    data_frame.insert(2, 'Coherence wrt context', coherence_array, True)
    return data_frame


# Analyzes a chatters' responses, assessing whether or not they are coherent with the given input.
def MLI3TC1(conv_array, data_frame):
    # Array for collecting the score
    print("     MLI3TC1")
    nsp_points = []

    for index in range(1, len(conv_array), 2):
        relevant_conv_array = util_functions.check_length_str_array(conv_array[0:(index - 1)], 512)

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(
                len(relevant_conv_array) - 1)]])  # conv_array[0:(index - 1)]])
        chatter_response = conv_array[index]

        # Setting up the tokenizer
        # inputs = bert_tokenizer(conv_string_input, chatter_response, return_tensors='pt')

        # Predicting the coherence score using Sentence-BERT
        # outputs = bert_model(**inputs)
        temp_list = util_functions.nsp(conv_string_input, chatter_response)

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    coherence_array = util_functions.judge_coherences(nsp_points, 2)
    data_frame.insert(2, 'Coherence wrt context', coherence_array, True)
    return data_frame


# Checks the max amount of duplicate ngrams for each length and returns the stutter degree,
# which is the mean amount of stutter words for all ngrams.
def MLA6TC1(conv_array, data_frame):
    print("     MLA6TC1")
    results = []
    for sentence in conv_array:
        sentencearray = list(sentence.split())
        n = len(sentencearray)

        # If the sentence only has length 1, break
        if n == 1:
            results.append(0)
            continue

        # Preallocate
        maxvals = [None] * (n - 2)

        # Find the most repeated gram of each length
        for order in range(2, n):
            grams = Counter(ngrams(sentencearray, order))
            # maxkeys[order - 1] = max(grams, key=grams.get)
            maxvals[order - 2] = max(grams.values())

        # Evaluate stutter
        # Amount of stutter is mean amount of stutter words for all ngrams
        results.append(sum([(maxvals[i - 2] - 1) * i / n for i in range(2, n)]))

    # Insert data
    if show_detailed:
        data_frame.insert(2, "MLA6TC1 (detailed)", results, True)

    if show_binary:
        bin_results = util_functions.threshold(results, False, thresh=0.33)
        data_frame.insert(2, "MLA6TC1", bin_results, True)
    return data_frame


# Method for assessing the toxicity-levels of any text input, a text-array of any size
def MLP1TC1(text, data_frame):
    print("     MLP1TC1")
    # The model takes in one or several strings
    results = model.predict(text)  # Assessment of several strings

    if show_binary:
        # Binary results, Pass or Fail
        bin_results = {}
        for col in results:
            bin_results[col] = util_functions.threshold(results[col], False, thresh=0.1, approve_above_threshold=False)
        df_bin_results = pd.DataFrame(data=bin_results).round(5)  # Presents the data as a Panda-Dataframe
        data_frame = pd.concat([data_frame, df_bin_results], axis=1)  # Adds the results to the data frame

    if show_detailed:
        df_results = pd.DataFrame(data=results).round(5)  # Presents the data as a Panda-Dataframe
        for col in results:
            df_results[col + " (detailed)"] = df_results.pop(col)
        data_frame = pd.concat([data_frame, df_results], axis=1)  # Adds the results to the data frame
    return data_frame


# Method for assessing whether any question is repeated at an abnormal frequency
def analyze_question_freq(conv_array, data_frame):
    print("     Question frequency")
    # The question vocabulary with corresponding frequencies
    question_vocab = []

    # Builds up the question vocabulary with all questions appearing in conv_array
    extracted_questions = util_functions.extract_question(conv_array)

    for sent in extracted_questions:
        # Adds the question to the vocab
        question_vocab.append(sent)

    # Counts the frequency per question and stores everything in a dictionary, mapping questions to their frequencies
    question_vocab = Counter(question_vocab)

    questions_repeated = []

    # Initiates questions_repeated to be an array of 'False's. Then per sentence, the question is extracted and checked
    # whether the sentence has a frequency > 1. If a question contains a question that has a frequency > 1, the index of
    # that question is set to 'True'.
    for index in range(len(conv_array)):
        questions_repeated.append('Pass')
        extracted_questions = util_functions.extract_question([conv_array[index]])

        for ex_quest in extracted_questions:
            if ex_quest in question_vocab:
                if question_vocab[ex_quest] > 1:
                    questions_repeated[index] = 'Fail'

    # Inserts the questions_repeated array into the data_frame.
    data_frame.insert(2, "rep_q", questions_repeated, True)

    return data_frame


# Analyzes the time taken for a chatter to respond and classifies it using three time intervals
def analyze_times(data_frame, time_array):
    # Inserts the time assessment of every response took into Chatter's data_frame
    if data_frame is None:
        data_frame = pd.DataFrame(data=time_array, columns=['Response times'])
    else:
        data_frame["Response times"] = time_array
    return data_frame


# Test case for analyzing how much the chatbot may remember information for a long time.
def MLI1TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI1TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1010000 + test_set["id"] + 0.5)

        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI1TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI1TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI1TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def MLI4TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI4TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1040000 + test_set["id"] + 0.5)

        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI4TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI4TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI4TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand questions formulated in several ways.
def MLI5TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI5TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1050000 + test_set["id"] + 0.5)

        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI5TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI5TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI5TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def MLI6TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI6TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1060000 + test_set["id"] + 0.5)
        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI6TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI6TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI6TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def MLI7TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI7TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1070000 + test_set["id"] + 0.5)
        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI7TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI7TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI7TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Analyzes whether Emely is consistent with its own information
def MLI13TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI13TC1")

    for test_set in test_sets:
        # Extract the answers
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1130000 + test_set["id"])

        if len(answers) > 0:
            # Separate the test whether it is a directed question or not
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([interpret[0]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)
            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == interpret[0] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI13TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI13TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "MLI13TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU3TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU3TC1")
    test_case = 'MLU3TC1'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many word order swaps that can be made while the chatbot still understands and answers
# properly.
def MLU4TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU4TC1")
    test_case = 'MLU4TC1'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU5TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU5TC1")
    test_case = 'MLU5TC1'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU6TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU6TC1")
    test_case = 'MLU6TC1'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame
