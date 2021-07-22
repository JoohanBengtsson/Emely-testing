# It might be a good idea to make some variables global, such as test_ids
import MLI13TC1

# General
import random
import time
import pandas as pd
import math
from numpy import cumsum

# Generate conversation specific
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import requests

# Affective text generator specific
import sys
from os import path

import config

sys.path.append(path.abspath("affectivetextgenerator"))
from affectivetextgenerator.run import generate

# Own script files
import util_functions  # Utility functions
import test_functions  # Test functions for analysis
from config import *  # Settings


# --------------------------- Functions ---------------------------

def init_tests():
    test_sets = {} # Which test sets will be used?
    test_ids = [0] * conversation_length # Which tests will be run?

    # Set test sets for the run
    if p_MLI4TC1 > 0:
        test_sets["MLI4TC1"] = assign_dataset("MLI4TC1", maxsets_MLI4TC1)
        # Assign random test set
#        nsets = random.randint(1, maxsets_MLI4TC1)  # Random number of test sets
#        r = random.sample(range(MLI13TC1.general["n_QA"]), k = nsets) # Random sequence
#        sets = [0]*nsets
#        for i in range(nsets): # 0 to maxsets - 1
#            setname = "ds" + str(MLI13TC1.general["MLI4TC1"] + r[i])
#            sets[i] = getattr(MLI13TC1,setname)
#        test_sets["MLI4TC1"] = sets
            #test_set = "ds{test_ids}".format(test_ids = random_id)
            #test_sets["MLI4TC1"] = getattr(MLI4TC1, test_set)

    if p_MLI13TC1 > 0:
        test_sets["MLI13TC1"] = assign_dataset("MLI13TC1", maxsets_MLI13TC1)
        # Assign random test set
#        nsets = random.randint(1, maxsets_MLI13TC1)  # Random number of test sets
#        r = random.sample(range(MLI13TC1.general["n_CO"]), k=nsets)  # Random sequence
#        sets = [0] * nsets
#        for i in range(nsets):  # 0 to maxsets - 1
#            setname = "ds" + str(MLI13TC1.general["MLI13TC1"] + r[i])
#            sets[i] = getattr(MLI13TC1,setname)
#        test_sets["MLI13TC1"] = sets
        # test_set = "ds{test_ids}".format(test_ids = random_id)
        # test_sets["MLI4TC1"] = getattr(MLI4TC1, test_set)

    cum_probability = list(cumsum([p_MLI4TC1, p_MLI13TC1])) # Last element shall not be greater than 1
    # Set indices for tests
    for i in range(1, conversation_length):
        if test_ids[i] == 0:
            u = random.uniform(0, 1)
            if u < cum_probability[0] and i < conversation_length - 2:
                # MLI4TC1
                test_id = 1040000 + random.choice([ts["id"] for ts in test_sets["MLI4TC1"]])
                test_ids[i] = test_id # The information
                test_ids[i + 1] = test_id + 0.5  # The question
            elif u < cum_probability[1] and i < conversation_length - 4:
                # MLI13TC1
                # Choose randomly from the ones that only requires one index
                test_ids[i] = 1130000 + random.choice([ts["id"] for ts in test_sets["MLI13TC1"]])
    return test_sets, test_ids


def assign_dataset(testname, maxsets):
    testtype = MLI13TC1.general[testname]
    # Assign random test set
    nsets = random.randint(1, maxsets)  # Random number of test sets
    r = random.sample(range(MLI13TC1.general["n_"+testtype]), k=nsets)  # Random sequence
    sets = [0] * nsets
    for i in range(nsets):  # 0 to maxsets - 1
        setname = "ds" + str(MLI13TC1.general[testtype] + r[i])
        sets[i] = getattr(MLI13TC1, setname)
    return sets

# Method for loading a conversation from a .txt
def load_conversation():
    text_file = open(load_document, 'r')  # Load a text. Split for each newline \n
    text = text_file.read()
    convarray = text.split('\n')
    # conversation_length = int(len(convarray) / 2)  # Length of convarray must be even. Try/catch here?
    # print(conversation_length)
    text_file.close()
    return convarray


# Method for saving a document to .txt
def save_conversation(save_conv_document, convarray):
    # Save the entire conversation
    convstring = util_functions.array2string(convarray)
    with open("saved_conversations/" + save_conv_document, 'w') as f:
        f.write(convstring)


# Method for looping conversation_length times, generating the conversation.
def generate_conversation():
    model_chatter1 = assign_model(1)
    model_chatter2 = assign_model(2)
    conv_array = []
    # chatter1_time = 0

    # The variable init_conv_randomly decides whether or not to initiate the conversation randomly.
    if init_conv_randomly:
        random_conv_starter()
        chatter1_times.append('-')
        chatter2_times.append('-')

    # Loop a conversation for an amount of conversation_length rounds, minus the rows if predefined on forehand.
    while len(conv_array) < 2 * conversation_length:
        conv_array = generate_conversation_step(model_chatter1, model_chatter2)

    if is_save_conversation:
        save_conversation(save_conv_document, conv_array)

    # print(str(chatters[0]) + " time: {:.2f}s".format(chatter1_time))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return conv_array


def generate_conversation_step(model_chatter1, model_chatter2):
    # So that the global variable convarray becomes directly available
    global convarray

    # Generates a response from chatter2, appends the response to convarray and prints the response
    t_start = time.time()
    test_id = test_ids[int(math.ceil((len(convarray) - 1) / 2))]  # int(math.ceil((6-1)/2))
    test_type = int(test_id/10000) # The test set
    test_ds = test_id%10000 # The test dataframe
    if test_type == 104 and test_ds%1 == 0: #and test_id%10000 in [t["id"] for t in test_sets["MLI4TC1"]]:
        test_set = getattr(MLI13TC1, "ds" + str(test_ds))
        resp = random.choice(test_set["information"])
    elif test_type == 104 and test_ds%1 == 0.5: #and test_id%10000 in [t["id"]+0.5 for t in test_sets["MLI4TC1"]]:
        test_set = getattr(MLI13TC1, "ds" + str(int(test_ds)))
        resp = random.choice(test_set["question"])
    elif test_type == 113: #and test_id%10000 in [t["id"] for t in test_sets["MLI13TC1"]]:
        test_set = getattr(MLI13TC1, "ds" + str(test_ds))
        resp = random.choice(test_set["words"])
    else:
        resp = model_chatter1.get_response(convarray)
    chatter1_times.append(time.time() - t_start)
    convarray.append(resp)
    print(str(chatters[0]) + ": ", resp)

    # Generates a response from chatter1, appends the response to convarray and prints the response
    t_start = time.time()
    resp = model_chatter2.get_response(convarray)
    chatter2_times.append(time.time() - t_start)
    convarray.append(resp)
    print(str(chatters[1]) + ": ", resp)

    return convarray


def random_conv_starter():
    # Chatter1 initiates with a greeting.
    convarray.append('Hey')
    print(str(chatters[0]) + ': Hey')

    if is_affect:
        # Generate a sentence from the affect model
        conv_start_resp = generate("You are a", topic, affect, knob)
        conv_start_resp = conv_start_resp[0][len('<|endoftext|>'):]
    else:
        # Pipeline for a random starter phrase for chatter2 in the conversation.
        text_gen = pipeline('text-generation')
        conv_start_resp = text_gen('I', max_length=50)[0]['generated_text']  # , do_sample=False))
        # Shortens the sentence to be the first part, if the sentence has any sub-sentences ending with a '.'.
        if '.' in conv_start_resp:
            conv_start_resp = conv_start_resp.split('.')[0]
    print(str(chatters[1]) + ": " + conv_start_resp)
    convarray.append(conv_start_resp)


# Assigns the chatter profile to any chatter. In order to extend to other chatters, classes need to be created and this
# function needs to be updated correspondingly.
def assign_model(nbr):
    chatter_profile = chatters[nbr - 1]
    if chatter_profile == 'emely':
        return Emely()
    elif chatter_profile == 'blenderbot':
        return BlenderBot()
    elif chatter_profile == 'user':
        return User()
    elif chatter_profile == 'predefined':
        return Predefined(nbr)


# Analyzes the conversation
def analyze_conversation(conv_array, test_sets, chatter1_times, chatter2_times):
    # Define variables
    data_frame_input = pd.DataFrame()
    data_frame = pd.DataFrame()
    conv_chatter1 = []
    conv_chatter2 = []

    # Separating convarray to the two chatter's respective conversation arrays
    for index in range(len(conv_array)):
        if index % 2 == 0:
            conv_chatter1.append(conv_array[index])
        else:
            conv_chatter2.append(conv_array[index])

    data_frame_input.insert(0, "Conversation", conv_chatter1)
    data_frame.insert(0, "Conversation", conv_chatter2)

    if is_MLP1TC1:
        # Analyze the two conversation arrays separately for toxicity and store the metrics using dataframes.
        data_frame_input = test_functions.MLP1TC1(conv_chatter1,
                                                  data_frame_input)  # analyze_word(conv_chatter2, data_frame_input)
        data_frame = test_functions.MLP1TC1(conv_chatter2, data_frame)  # analyze_word(conv_chatter1, data_frame)

    if is_MLI2TC1:
        # Check responses to see how likely they are to be coherent ones w.r.t the context.
        # Here the entire conversation array needs to be added due to the coherence test design
        data_frame_input = test_functions.MLI2TC1(conv_array, data_frame_input, 1)  # Context
        data_frame = test_functions.MLI2TC1(conv_array, data_frame, 2)  # Context

    if is_MLI3TC1:
        # Check responses to see how likely they are to be coherent ones w.r.t the input.
        # Here the entire conversation array needs to be added due to the coherence test design
        data_frame_input = test_functions.MLI3TC1(conv_array, data_frame_input, 1)  # Last answer
        data_frame = test_functions.MLI3TC1(conv_array, data_frame, 2)  # Last answer

    if is_analyze_question_freq:
        # Check for recurring questions and add metric to dataframe
        test_functions.analyze_question_freq(conv_chatter1, data_frame_input)
        test_functions.analyze_question_freq(conv_chatter2, data_frame)

    if is_MLA6TC1:
        # Check for stuttering using N-grams, and add metric to dataframe
        data_frame_input = test_functions.MLA6TC1(conv_chatter1, data_frame_input)
        data_frame = test_functions.MLA6TC1(conv_chatter2, data_frame)

    if p_MLI13TC1 > 0 and is_load_conversation == False:
        data_frame = test_functions.MLI13TC1(data_frame, conv_chatter2, test_ids, test_sets["MLI13TC1"])
        # data_frame = test_functions.MLI13TC2(data_frame, conv_chatter1, test_sets)

    if p_MLI4TC1 > 0 and is_load_conversation == False:
        data_frame = test_functions.MLI4TC1(data_frame, conv_chatter2, test_ids, test_sets["MLI4TC1"])

    if not is_load_conversation:
        data_frame_input = test_functions.analyze_times(data_frame_input, chatter1_times)
        data_frame = test_functions.analyze_times(data_frame, chatter2_times)

    global df_summary, df_input_summary
    df_summary = pd.concat([df_summary, data_frame], axis=1)
    df_input_summary = pd.concat([df_input_summary, data_frame_input], axis=1)
    return data_frame, data_frame_input


# Prints every row of the data_frame collecting all metrics. Writes to a Excel-file
def write_to_excel(df, name):
    df.to_excel("./reports/" + name + '_report.xlsx')


# --------------------------- Classes ---------------------------


# Here the chatter profiles are defined. In order to extend to more chatters, a class needs to be defined here and the
# get_response method must be implemented.

class BlenderBot:
    def __init__(self):
        self.name = 'facebook/blenderbot-400M-distill'
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.name)

    def get_response(self, conv_array):
        conv_string = self.__array2blenderstring(conv_array[-prev_conv_memory_chatter2:])
        inputs = self.tokenizer([conv_string], return_tensors='pt')
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response

    def __array2blenderstring(self, conv_array):
        conv_string = ' '.join([str(elem) + '</s> <s>' for elem in conv_array[-3:]])
        conv_string = conv_string[:len(conv_string) - 8]
        return conv_string


class Emely:
    def __init__(self):
        self.URL = "http://localhost:8080/inference"

    def get_response(self, conv_array):
        # Inputs the conversation array and outputs a response from Emely
        json_obj = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "text": util_functions.array2string(conv_array[-prev_conv_memory_chatter1:])
        }
        r = requests.post(self.URL, json=json_obj)
        response = r.json()['text']
        if len(response) > 128:
            response = response[0:128]
        return response


class User:
    def __init__(self):
        global init_conv_randomly
        init_conv_randomly = False

    def get_response(self, convarray):
        user_input = input("Write your response: ")
        return user_input


class Predefined:

    def __init__(self, nbr):
        if nbr == 1:
            self.predefined_conv = predefined_conv_chatter1
        else:
            self.predefined_conv = predefined_conv_chatter2

        global conversation_length
        global init_conv_randomly

        conversation_length = len(self.predefined_conv)

        init_conv_randomly = False

    def get_response(self, convarray):
        return self.predefined_conv.pop(0)


# --------------------------- Main-method ---------------------------
if __name__ == '__main__':
    script_start_time = time.time()
    # Data frames containing all the data frames collected from each conversation per chatter
    df_summary = pd.DataFrame()  # Data frame containing all the data frames collected from each conversation
    df_input_summary = pd.DataFrame()  # Data frame containing all the data frames collected from each conversation from the chatter2

    # Initialize tests by defining where the tests will be.
    test_sets, test_ids = init_tests()
    for run in range(max_runs):
        # Define variables
        convarray = convarray_init
        chatter1_times = []
        chatter2_times = []

        print('Starting conversation ' + str(run + 1))
        start_time = time.time()

        if not is_load_conversation:
            # Load conversation
            print("Generating conversation...")
            convarray = generate_conversation()
        else:
            print("Loading conversation...")
            convarray = load_conversation()

        if is_analyze_conversation:
            # Starts the analysis of the conversation
            print("Analyzing conversation...")
            df_1, df_2 = analyze_conversation(convarray, test_sets, chatter1_times, chatter2_times)
            print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # The method for presenting the metrics into a .xlsx-file. Will print both the summary-Dataframes to .xlsx
    print("Exporting results...")
    write_to_excel(df_1, save_analysis_names[0])
    write_to_excel(df_2, save_analysis_names[1])

    print("Done!")
    print('Total time the script took was: ' + str(time.time() - script_start_time) + 's')
