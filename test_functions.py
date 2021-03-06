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

import csv

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

# Test case for REQ_P2: Assessing the toxicity-levels of any text input, a text-array of any size
def TC_REQ_P2(text, data_frame, folder, run_ID):
    print("     TC_REQ_P2")
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

    if print_distributions:
        toxicity_results = {}
        for col in results:
            toxicity_results[col] = (results[col])

        tmp = folder + "REQ_P2_runID-" + str(run_ID)
        file = open(tmp + ".csv", "w")
        for key, list in toxicity_results.items():
            file.write(key + ",")
            count = 0
            for item in list:
                count += 1
                file.write(str(item))
                if (count < len(list)):
                    file.write(",")
            file.write("\n")
        file.close()

        pd.read_csv(tmp + ".csv", header=None).T.to_csv(tmp + "v2.csv", header=False, index=False)

    return data_frame

# Analyzes a chatter's reply, assessing whether or not they are coherent with the current dialog
def TC_REQ_I2(conv_array, data_frame, folder, run_ID):
    # Array for collecting the score
    print("     TC_REQ_I3")
    nsp_points = []
    coherence_exceptions = []

    for index in range(1, len(conv_array), 2):
        relevant_conv_array = util_functions.check_length_str_array(conv_array[0:(index - 1)], 512)

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(
                len(relevant_conv_array) - 1)]])  # conv_array[0:(index - 1)]])
        chatter_response = conv_array[index]

        # Setting up the tokenizer
        # inputs = bert_tokenizer(conv_string_input, chatter_response, return_tensors='pt')

        # Predicting the coherence score using Sentence-BERT
        # outputs = bert_model(**inputs)
        #print("I3 analysis of: " + conv_string_input)
        temp_list = util_functions.nsp(conv_string_input, chatter_response)

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

        if (temp_list[0] < temp_list[1]):
            coherence_exceptions.append((chatter_response, temp_list[0], temp_list[1]))

    if (print_distributions):
        tmp = folder + "REQ_I3_runID-" + str(run_ID) # consider them I2

        if (len(coherence_exceptions) > 0):
            file = open(tmp + ".csv", "w")
            for item in coherence_exceptions:
                file.write(str(item))
                file.write("\n")
            file.close()

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    # coherence_array = util_functions.judge_coherences(nsp_points, 2)
    coherence_array = util_functions.threshold(nsp_points, False, thresh=-6)
    data_frame.insert(2, 'TC_REQ_I3', coherence_array, True)
    return data_frame

# Analyzes a chatter's reply, assessing whether or not they are coherent with the given prompt
def TC_REQ_I3(conv_array, data_frame):
    # Array for collecting the score
    print("     TC_REQ_I2")
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
        #print("I2 analysis of: " + conv_string_input)
        temp_list = util_functions.nsp(conv_string_input, chatter_response)

        # Calculating the difference between tensor(0) indicating the grade of coherence, and tensor(1) indicating the
        # grade of incoherence
        nsp_points.append(temp_list[0] - temp_list[1])

    # Using judge_coherences to assess and classify the points achieved from Sent-BERT
    coherence_array = util_functions.judge_coherences(nsp_points, 2)
    data_frame.insert(2, 'TC_REQ_I2', coherence_array, True)
    return data_frame

# Test case for REQ-A4: Checks the max amount of duplicate ngrams for each length and returns the stutter degree,
# which is the mean amount of stutter words for all ngrams.
def TC_REQ_A4(conv_array, data_frame, folder, run_ID):
    print("     TC_REQ_A4")
    results = []
    replies = []
    stuttering_results = []

    for sentence in conv_array:
        replies.append(sentence.replace(",", " "))
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
        stutter_score = sum([(maxvals[i - 2] - 1) * i / n for i in range(2, n)])
        results.append(stutter_score)
        stuttering_results.append(stutter_score)

    # Insert data
    if show_detailed:
        data_frame.insert(2, "TC_REQ_A4 (detailed)", results, True)

    if (print_distributions):
        tmp = folder + "REQ_A4_runID-" + str(run_ID)
        file = open(tmp + ".csv", "w")
        index = 0
        for item in stuttering_results:

            file.write(replies[index] + "," + str(item))
            file.write("\n")
            index += 1
        file.close()

    if show_binary:
        bin_results = util_functions.threshold(results, False, thresh=0.33, approve_above_threshold=False)
        data_frame.insert(2, "TC_REQ_A4", bin_results, True)
    return data_frame


# Test case for REQ-A3: Assessing whether any question is repeated at an abnormal frequency
def TC_REQ_A3(conv_array, data_frame, folder, run_ID):
    print("     TC_REQ_A3")
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

    print("QREP verdict: " + str(questions_repeated))

    # Inserts the questions_repeated array into the data_frame.
    data_frame.insert(2, "TC_REQ_A3", questions_repeated, True)

    if (print_distributions):
        tmp = folder + "REQ_A3_runID-" + str(run_ID)
        file = open(tmp + ".csv", "w")
        for key, item in question_vocab.items():
            file.write(key + "," + str(item))
            file.write("\n")
        file.close()

    return data_frame


# Analyzes the time taken for a chatter to respond and classifies it using three time intervals
def analyze_times(data_frame, time_array):
    # Inserts the time assessment of every response took into Chatter's data_frame
    if data_frame is None:
        data_frame = pd.DataFrame(data=time_array, columns=['Response times'])
    else:
        data_frame["Response times"] = time_array
    return data_frame


# Test case for REQ-I5: Analyzing to what extent the chatbot remembers information for a long time.
def TC_REQ_I5(data_frame, conv_chatter, test_ids, test_sets, folder, run_ID):
    print("     TC_REQ_I5")

    check_memory = []

    for test_set in test_sets:

        # Extract the answers only given after the question
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1010000 + test_set["id"] + 0.5)
        print("Number of answers: " + str(len(answers)))

        if len(answers) > 0:
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                results = util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)
                check_memory.append("Expected result: " + str([test_set["answer"]]))
                check_memory.append("Observed results: " + str(interpret))
                check_memory.append("Test verdicts: " + str(bin_results))
                print("Found a " + str(interpret) + " that was a " + str(bin_results))
            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                results = [i == test_set["answer"] for i in interpret]
                bin_results = util_functions.threshold(results, True)
                check_memory.append("Expected result: " + str([test_set["answer"]]))
                check_memory.append("Observed results: " + str(interpret))
                check_memory.append("Test verdicts: " + str(bin_results))
                print("Found a " + str(interpret) + " that was a " + str(bin_results))

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I5 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I5 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I5 - " + str(test_set["id"]), bin_results)

    if (print_distributions):
        tmp = folder + "REQ_I5_runID-" + str(run_ID)

        if (len(check_memory) > 0):
            file = open(tmp + ".csv", "w")
            for item in check_memory:
                file.write(str(item))
                file.write("\n")
            file.close()


    return data_frame


# Test case for REQ-I8: analyzing how robustly the chatbot understands information provided in different ways.
def TC_REQ_I8(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_I8")
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
                data_frame.insert(2, "TC_REQ_I8 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I8 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I8 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for REQ-I10: analyzing how robustly the chatbot may understand questions formulated in different ways.
def TC_REQ_I10(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_I10")
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
                data_frame.insert(2, "TC_REQ_I10 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I10 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I10 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def TC_REQ_I9(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_I9")
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
                data_frame.insert(2, "TC_REQ_I9 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I9 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I9 - " + str(test_set["id"]), bin_results)
    return data_frame


# Test case for analyzing how much the chatbot may understand sentences formulated in several ways.
def TC_REQ_I11(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_I11")
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
                data_frame.insert(2, "TC_REQ_I11 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I11 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I11 - " + str(test_set["id"]), bin_results)
    return data_frame


# Analyzes whether the chatbot is consistent with its own information
def TC_REQ_I1(data_frame, conv_chatter, test_ids, test_sets, folder, run_ID):
    print("     TC_REQ_I1")

    for test_set in test_sets:
        # Extract the answers
        answers, test_idx = util_functions.extract_answers(conv_chatter, test_ids, 1130000 + test_set["id"])

        info_consistency = []

        if len(answers) > 0:
            # Separate the test whether it is a directed question or not
            if not test_set["directed"]:
                # Reduce the answer to the specific answer to the question.
                interpret = util_functions.openQA(answers, test_set["QA"])
                info_consistency.append("Expected result: " + str([interpret[0]] * len(interpret)))
                info_consistency.append("Observed results: " + str(interpret))
                results = util_functions.check_similarity([interpret[0]] * len(interpret), interpret)
                bin_results = util_functions.threshold(results, False, thresh=config.threshold_sem_sim_tests)
                print(bin_results)
                info_consistency.append("Test verdicts (skip first): " + str(bin_results))
            else:
                # Check whether the answer is true
                interpret = util_functions.binaryQA(answers)
                info_consistency.append("Expected results: " + str([interpret[0]]))
                info_consistency.append("Observed results: " + str(interpret))
                results = [i == interpret[0] for i in interpret]
                bin_results = util_functions.threshold(results, True)
                print(bin_results)
                info_consistency.append("Test verdicts (skip first): " + str(bin_results))

            # Add the results to the data frame. Rows outside of the test gets the value None
            if show_interpret:
                interpret = util_functions.create_column(interpret, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I1 (interpret) - " + str(test_set["id"]), interpret)

            if show_detailed:
                results = util_functions.create_column(results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I1 (detailed) - " + str(test_set["id"]), results)

            if show_binary:
                bin_results = util_functions.create_column(bin_results, test_idx, len(conv_chatter))
                data_frame.insert(2, "TC_REQ_I1 - " + str(test_set["id"]), bin_results)

            if (print_distributions):
                tmp = folder + "REQ_I1_runID-" + str(run_ID)

                if (len(info_consistency) > 0):
                    file = open(tmp + ".csv", "w")
                    for item in info_consistency:
                        file.write(str(item))
                        file.write("\n")
                    file.close()
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def TC_REQ_U3(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_U3")
    test_case = 'TC_REQ_U3'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many word order swaps that can be made while the chatbot still understands and answers
# properly.
def TC_REQ_U4(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_U4")
    test_case = 'TC_REQ_U4'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def TC_REQ_U5(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_U5")
    test_case = 'TC_REQ_U5'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame


# Test case for testing how many typing mistakes can be made while the chatbot still understands and answers properly.
def TC_REQ_U6(data_frame, conv_chatter, test_ids, test_sets):
    print("     TC_REQ_U6")
    test_case = 'TC_REQ_U6'
    data_frame = util_functions.ux_test_analysis(data_frame=data_frame, conv_chatter=conv_chatter, test_ids=test_ids,
                                                 test_sets=test_sets, test_case=test_case)
    return data_frame
