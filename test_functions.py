# Test functions are the functions used for assessing a conversation.
# They are linked to the requirements presented in the SRS.

import requests
import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertForNextSentencePrediction #BlenderbotConfig, pipeline, \
from detoxify import Detoxify
from collections import Counter
from nltk import ngrams
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

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(len(relevant_conv_array)-1)]])#conv_array[0:(index - 1)]])
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

        conv_string_input = ' '.join([str(elem) + ". " for elem in relevant_conv_array[0:(len(relevant_conv_array)-1)]])#conv_array[0:(index - 1)]])
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
            #maxkeys[order - 1] = max(grams, key=grams.get)
            maxvals[order - 2] = max(grams.values())

        # Evaluate stutter
        # Amount of stutter is mean amount of stutter words for all ngrams
        stutterval.append(sum([(maxvals[i-2]-1)*i/n for i in range(2, n)]))

    # Insert data
    data_frame.insert(1, "stutter", stutterval, True)
    return data_frame


# Method for assessing the toxicity-levels of any text input, a text-array of any size
def MLP1TC1(text, data_frame):
    print("     MLP1TC1")
    # The model takes in one or several strings
    results = model.predict(text) # Assessment of several strings
    df_results = pd.DataFrame(data=results).round(5) # Presents the data as a Panda-Dataframe
    data_frame = pd.concat([data_frame, df_results], axis=1) # Adds the results to the data frame
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


# Analyzes whether Emely is consistent with its own information
def MLI13TC1(data_frame, conv_chatter, idx_MLI13TC1):
    print("     MLI13TC1")
    # Extract the answers and judge their similarity
    answers = [conv_chatter[int((i + 1)/2)] for i in idx_MLI13TC1]
    results = util_functions.check_similarity([answers[0] for i in answers], answers)

    # Add the results to the data frame. Rows outside of the test gets the value 0
    consistency = [0] * len(conv_chatter)
    for i in range(len(idx_MLI13TC1)):
        consistency[int((idx_MLI13TC1[i] + 1) / 2)] = results[i]
    data_frame.insert(1, "Consistency", consistency)
    return data_frame
