# Test functions are the functions used for assessing a conversation.
# They are linked to the requirements presented in the SRS.

import requests
import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertForNextSentencePrediction  # BlenderbotConfig, pipeline, \
from detoxify import Detoxify
from collections import Counter
from nltk import ngrams

from config import *
import main
import util_functions

# --------------------------- External modules ---------------------------
# These rather slow-loaded models are not loaded if present_metrics is not true, to reduce the startup time when working
# on the code.
# Initiates Bert for Next Sentence Prediction (NSP) and stores the result
bert_type = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
bert_model = BertForNextSentencePrediction.from_pretrained(bert_type)

# To specify the device the Detoxify-model will be allocated on (defaults to cpu), accepts any torch.device input
if torch.cuda.is_available():
    model = Detoxify('unbiased', device='cuda')
else:
    model = Detoxify('unbiased')


# --------------------------- Test functions ---------------------------


# Analyzes responses of chatter number chatter_index w.r.t the whole conversation that has passed.
def MLI2TC1(conv_array, data_frame, chatter_index):
    # Array for collecting the score
    print("     MLI2TC1")

    nsp_points = []

    for index in range(3 - chatter_index, len(conv_array), 2):
        relevant_conv_array = util_functions.check_length_str_array(conv_array[0:(index - 1)], 512)

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(
                len(relevant_conv_array) - 1)]])  # conv_array[0:(index - 1)]])
        chatter_response = conv_array[index]

        # Setting up the tokenizer
        inputs = bert_tokenizer(conv_string_input, chatter_response, return_tensors='pt')

        # Predicting the coherence score using Sentence-BERT
        outputs = bert_model(**inputs)
        temp_list = outputs.logits.tolist()[0]

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    coherence_array = util_functions.judge_coherences(nsp_points, chatter_index)
    data_frame.insert(1, 'Coherence wrt context', coherence_array, True)
    return data_frame


# Analyzes a chatters' responses, assessing whether or not they are coherent with the given input.
def MLI3TC1(conv_array, data_frame, chatter_index):
    # Array for collecting the score
    print("     MLI3TC1")
    nsp_points = []

    for index in range(3 - chatter_index, len(conv_array), 2):
        relevant_conv_array = util_functions.check_length_str_array(conv_array[0:(index - 1)], 512)

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(
                len(relevant_conv_array) - 1)]])  # conv_array[0:(index - 1)]])
        chatter_response = conv_array[index]

        # Setting up the tokenizer
        inputs = bert_tokenizer(conv_string_input, chatter_response, return_tensors='pt')

        # Predicting the coherence score using Sentence-BERT
        outputs = bert_model(**inputs)
        temp_list = outputs.logits.tolist()[0]

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    coherence_array = util_functions.judge_coherences(nsp_points, chatter_index)
    data_frame.insert(1, 'Coherence wrt context', coherence_array, True)
    return data_frame


# Checks the max amount of duplicate ngrams for each length and returns the stutter degree,
# which is the mean amount of stutter words for all ngrams.
def MLA6TC1(conv_array, data_frame):
    print("     MLA6TC1")
    stutterval = []
    for sentence in conv_array:
        sentencearray = list(sentence.split())
        n = len(sentencearray)

        # If the sentence only has length 1, break
        if n == 1:
            stutterval.append(0)
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
        stutterval.append(sum([(maxvals[i - 2] - 1) * i / n for i in range(2, n)]))

    # Insert data
    data_frame.insert(1, "stutter", stutterval, True)
    return data_frame


# Method for assessing the toxicity-levels of any text input, a text-array of any size
def MLP1TC1(text, data_frame):
    print("     MLP1TC1")
    # The model takes in one or several strings
    results = model.predict(text)  # Assessment of several strings
    df_results = pd.DataFrame(data=results).round(5)  # Presents the data as a Panda-Dataframe
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
        questions_repeated.append('False')
        extracted_questions = util_functions.extract_question([conv_array[index]])

        for ex_quest in extracted_questions:
            if ex_quest in question_vocab:
                if question_vocab[ex_quest] > 1:
                    questions_repeated[index] = 'True'

    # Inserts the questions_repeated array into the data_frame.
    data_frame.insert(1, "rep_q", questions_repeated, True)

    return data_frame


# Analyzes the time taken for a chatter to respond and classifies it using three time intervals
def analyze_times(data_frame, time_array):
    #    time_assessment_array = []
    #
    #    for time_sample in time_array:
    #        if time_sample == '-':
    #            time_assessment_array.append(time_sample)
    #        elif time_sample <= 1:
    #            time_assessment_array.append('Great response time')
    #        elif time_sample <= 2:
    #            time_assessment_array.append('Good response time')
    #        elif time_sample > 2:
    #            time_assessment_array.append('Bad response time')
    #        else:
    #            time_assessment_array.append('-')

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
                bin_results = util_functions.threshold(results, False, thresh=0.3)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI1TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI1TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI1TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def MLI4TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI4TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1040000 + test_set["id"] + 0.5)

        if len(answers)>0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]]*len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=0.3)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI4TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI4TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI4TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand questions formulated in several ways.
def MLI5TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLI5TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1050000 + test_set["id"] + 0.5)

        if len(answers)>0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]]*len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=0.3)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI5TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI5TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI5TC1 - " + str(test_set["id"]), bin_results)
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
                results = util_functions.check_similarity([test_set["answer"]]*len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=0.3)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI6TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI6TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI6TC1 - " + str(test_set["id"]), bin_results)
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
                results = util_functions.check_similarity([test_set["answer"]]*len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=0.3)

            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI7TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI7TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI7TC1 - " + str(test_set["id"]), bin_results)
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
                results = util_functions.check_similarity([interpret[0]]*len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=0.3)
            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == interpret[0] for i in interpret]
                bin_results = util_functions.threshold(results, True)

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI13TC1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI13TC1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(1, "MLI13TC1 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU3TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU3TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 2030000 + test_set["id"] + 0.33)
        answers2, test_idx2 = util_functions.extract_answers(conv_chatter, test_ids, 2030000 + test_set["id"] + 0.66)
        answers3, test_idx3 = util_functions.extract_answers(conv_chatter, test_ids, 2030000 + test_set["id"] + 0.99)

        if not len(answers) > 0:
            continue

        if not test_set["directed"]:
            # Reduce the answer to the specific answer to the question.
            interpret = util_functions.openQA(answers, test_set["QA"])
            interpret2 = util_functions.openQA(answers2, test_set["QA"])
            interpret3 = util_functions.openQA(answers3, test_set["QA"])

            results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
            results2 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret2)
            results3 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret3)

            if show_binary:
                bin_results = util_functions.threshold(results, False, thresh=0.3)
                bin_results2 = util_functions.threshold(results2, False, thresh=0.3)
                bin_results3 = util_functions.threshold(results3, False, thresh=0.3)

        else:
            # Check whether the answer is true
            interpret = util_functions.binaryQA(answers)
            interpret2 = util_functions.binaryQA(answers2)
            interpret3 = util_functions.binaryQA(answers3)

            results = [i == test_set["answer"] for i in interpret]
            results2 = [i == test_set["answer"] for i in interpret2]
            results3 = [i == test_set["answer"] for i in interpret3]

            if show_binary:
                bin_results = util_functions.threshold(results, True)
                bin_results2 = util_functions.threshold(results2, True)
                bin_results3 = util_functions.threshold(results3, True)

        # Add the results to the data frame. Rows outside of the test gets the value None
        if show_interpret:
            interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
            interpret2 = util_functions.create_column(interpret2, test_idx2, len(conv_chatter))
            interpret3 = util_functions.create_column(interpret3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU3TC1 many typo's - " + str(test_set["id"]), interpret3)
            data_frame.insert(1, "MLU3TC1 some typo's - " + str(test_set["id"]), interpret2)
            data_frame.insert(1, "MLU3TC1 no typo - " + str(test_set["id"]), interpret)

        if show_detailed:
            results = util_functions.create_column(results, test_idx, len(conv_chatter))
            results2 = util_functions.create_column(results2, test_idx2, len(conv_chatter))
            results3 = util_functions.create_column(results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU3TC1 many typo's - " + str(test_set["id"]), results3)
            data_frame.insert(1, "MLU3TC1 some typo's - " + str(test_set["id"]), results2)
            data_frame.insert(1, "MLU3TC1 no typo - " + str(test_set["id"]), results)

        if show_binary:
            bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
            bin_results2 = util_functions.create_column(bin_results2, test_idx2, len(conv_chatter))
            bin_results3 = util_functions.create_column(bin_results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU3TC1 many typo's - " + str(test_set["id"]), bin_results3)
            data_frame.insert(1, "MLU3TC1 some typo's - " + str(test_set["id"]), bin_results2)
            data_frame.insert(1, "MLU3TC1 no typo - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU4TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU4TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 2040000 + test_set["id"] + 0.33)
        answers2, test_idx2 = util_functions.extract_answers(conv_chatter, test_ids, 2040000 + test_set["id"] + 0.66)
        answers3, test_idx3 = util_functions.extract_answers(conv_chatter, test_ids, 2040000 + test_set["id"] + 0.99)

        if not len(answers) > 0:
            continue

        if not test_set["directed"]:
            # Reduce the answer to the specific answer to the question.
            interpret = util_functions.openQA(answers, test_set["QA"])
            interpret2 = util_functions.openQA(answers2, test_set["QA"])
            interpret3 = util_functions.openQA(answers3, test_set["QA"])

            results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
            results2 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret2)
            results3 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret3)

            if show_binary:
                bin_results = util_functions.threshold(results, False, thresh=0.3)
                bin_results2 = util_functions.threshold(results2, False, thresh=0.3)
                bin_results3 = util_functions.threshold(results3, False, thresh=0.3)

        else:
            # Check whether the answer is true
            interpret = util_functions.binaryQA(answers)
            interpret2 = util_functions.binaryQA(answers2)
            interpret3 = util_functions.binaryQA(answers3)

            results = [i == test_set["answer"] for i in interpret]
            results2 = [i == test_set["answer"] for i in interpret2]
            results3 = [i == test_set["answer"] for i in interpret3]

            if show_binary:
                bin_results = util_functions.threshold(results, True)
                bin_results2 = util_functions.threshold(results2, True)
                bin_results3 = util_functions.threshold(results3, True)

        # Add the results to the data frame. Rows outside of the test gets the value None
        if show_interpret:
            interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
            interpret2 = util_functions.create_column(interpret2, test_idx2, len(conv_chatter))
            interpret3 = util_functions.create_column(interpret3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU4TC1 4 word order swaps - " + str(test_set["id"]), interpret3)
            data_frame.insert(1, "MLU4TC1 1 word order swaps - " + str(test_set["id"]), interpret2)
            data_frame.insert(1, "MLU4TC1 0 word order swaps - " + str(test_set["id"]), interpret)

        if show_detailed:
            results = util_functions.create_column(results, test_idx, len(conv_chatter))
            results2 = util_functions.create_column(results2, test_idx2, len(conv_chatter))
            results3 = util_functions.create_column(results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU4TC1 4 word order swaps - " + str(test_set["id"]), results3)
            data_frame.insert(1, "MLU4TC1 1 word order swaps - " + str(test_set["id"]), results2)
            data_frame.insert(1, "MLU4TC1 0 word order swaps - " + str(test_set["id"]), results)

        if show_binary:
            bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
            bin_results2 = util_functions.create_column(bin_results2, test_idx2, len(conv_chatter))
            bin_results3 = util_functions.create_column(bin_results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU4TC1 4 word order swaps - " + str(test_set["id"]), bin_results3)
            data_frame.insert(1, "MLU4TC1 1 word order swaps - " + str(test_set["id"]), bin_results2)
            data_frame.insert(1, "MLU4TC1 0 word order swaps - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def MLU5TC1(data_frame, conv_chatter, test_ids, test_sets):
    print("     MLU5TC1")
    for test_set in test_sets:
        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 2050000 + test_set["id"] + 0.33)
        answers2, test_idx2 = util_functions.extract_answers(conv_chatter, test_ids, 2050000 + test_set["id"] + 0.66)
        answers3, test_idx3 = util_functions.extract_answers(conv_chatter, test_ids, 2050000 + test_set["id"] + 0.99)

        if not len(answers) > 0:
            continue

        if not test_set["directed"]:
            # Reduce the answer to the specific answer to the question.
            interpret = util_functions.openQA(answers, test_set["QA"])
            interpret2 = util_functions.openQA(answers2, test_set["QA"])
            interpret3 = util_functions.openQA(answers3, test_set["QA"])

            results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
            results2 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret2)
            results3 = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret3)

            if show_binary:
                bin_results = util_functions.threshold(results, False, thresh=0.3)
                bin_results2 = util_functions.threshold(results2, False, thresh=0.3)
                bin_results3 = util_functions.threshold(results3, False, thresh=0.3)

        else:
            # Check whether the answer is true
            interpret = util_functions.binaryQA(answers)
            interpret2 = util_functions.binaryQA(answers2)
            interpret3 = util_functions.binaryQA(answers3)

            results = [i == test_set["answer"] for i in interpret]
            results2 = [i == test_set["answer"] for i in interpret2]
            results3 = [i == test_set["answer"] for i in interpret3]

            if show_binary:
                bin_results = util_functions.threshold(results, True)
                bin_results2 = util_functions.threshold(results2, True)
                bin_results3 = util_functions.threshold(results3, True)

        # Add the results to the data frame. Rows outside of the test gets the value None
        if show_interpret:
            interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
            interpret2 = util_functions.create_column(interpret2, test_idx2, len(conv_chatter))
            interpret3 = util_functions.create_column(interpret3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU5TC1 2 masked words - " + str(test_set["id"]), interpret3)
            data_frame.insert(1, "MLU5TC1 1 masked words - " + str(test_set["id"]), interpret2)
            data_frame.insert(1, "MLU5TC1 0 masked words - " + str(test_set["id"]), interpret)

        if show_detailed:
            results = util_functions.create_column(results, test_idx, len(conv_chatter))
            results2 = util_functions.create_column(results2, test_idx2, len(conv_chatter))
            results3 = util_functions.create_column(results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU5TC1 2 masked words - " + str(test_set["id"]), results3)
            data_frame.insert(1, "MLU5TC1 1 masked words - " + str(test_set["id"]), results2)
            data_frame.insert(1, "MLU5TC1 0 masked words - " + str(test_set["id"]), results)

        if show_binary:
            bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
            bin_results2 = util_functions.create_column(bin_results2, test_idx2, len(conv_chatter))
            bin_results3 = util_functions.create_column(bin_results3, test_idx3, len(conv_chatter))

            data_frame.insert(1, "MLU5TC1 2 masked words - " + str(test_set["id"]), bin_results3)
            data_frame.insert(1, "MLU5TC1 1 masked words - " + str(test_set["id"]), bin_results2)
            data_frame.insert(1, "MLU5TC1 0 masked words - " + str(test_set["id"]), bin_results)
    return data_frame
