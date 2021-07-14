# General
import time

# Generate conversation specific
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import requests

# Affective text generator specific
import sys
from os import path
sys.path.append(path.abspath("affectivetextgenerator"))
from affectivetextgenerator.run import generate

# Own script files
import util_functions  # Utility functions
import test_functions  # Test functions for analysis
from config import *  # Settings

# --------------------------- Functions ---------------------------


# Method for loading a conversation from a .txt
def load_conversation():
    text_file = open(load_document, 'r')  # Load a text. Split for each newline \n
    text = text_file.read()
    convarray = text.split('\n')
    #conversation_length = int(len(convarray) / 2)  # Length of convarray must be even. Try/catch here?
    #print(conversation_length)
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

    # The variable init_conv_randomly decides whether or not to initiate the conversation randomly.
    if init_conv_randomly:
        random_conv_starter()
        chatter1_times.append('-')
        chatter2_times.append('-')

    # Loop a conversation for an amount of conversation_length rounds, minus the rows if predefined on forehand.
    for i in range(conversation_length - int(len(convarray) / 2)):
        # Generates a response from chatters, appends the responses to convarray and prints the response. Also
        # records response times.
        generate_conversation_step(model_chatter1, model_chatter2)

    if is_save_conversation:
        save_conversation(save_conv_document, convarray)

    #print(str(chatters[0]) + " time: {:.2f}s".format(chatter1_time))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return convarray, chatter1_times, chatter2_times


def generate_conversation_step(model_chatter1, model_chatter2):
    t_start = time.time()
    resp = model_chatter1.get_response(convarray)
    chatter1_times.append(time.time() - t_start)
    convarray.append(resp)
    print(str(chatters[0]) + ": ", resp)

    t_start = time.time()
    resp = model_chatter2.get_response(convarray)
    chatter2_times.append(time.time() - t_start)
    convarray.append(resp)
    print(str(chatters[1]) + ": ", resp)


# Function for generating a random conversation starter
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
    chatter_profile = chatters[nbr-1]
    if chatter_profile == 'emely':
        return Emely()
    elif chatter_profile == 'blenderbot':
        return BlenderBot()
    elif chatter_profile == 'user':
        return User()
    elif chatter_profile == 'predefined':
        return Predefined(nbr)


# Analyzes the conversation
def analyze_conversation(conv_array, chatter1_times, chatter2_times):
    data_frame = None
    data_frame_input = None
    #df_summary = None  # Data frame containing all the data frames collected from each conversation
    #df_input_summary = None  # Data frame containing all the data frames collected from each conversation from the chatter2

    # Separating convarray to the two chatter's respective conversation arrays
    conv_chatter1 = []
    conv_chatter2 = []

    for index in range(len(conv_array)):
        if index % 2 == 0:
            conv_chatter1.append(conv_array[index])
        else:
            conv_chatter2.append(conv_array[index])

    if is_MLP1TC1:
        # Analyze the two conversation arrays separately for toxicity and store the metrics using dataframes.
        data_frame = test_functions.MLP1TC1(conv_chatter1, data_frame)#analyze_word(conv_chatter1, data_frame)
        data_frame_input = test_functions.MLP1TC1(conv_chatter2, data_frame_input)#analyze_word(conv_chatter2, data_frame_input)

    if is_MLI2TC1:
        # Check responses to see how likely they are to be coherent ones w.r.t the context.
        data_frame = test_functions.MLI2TC1(conv_array, data_frame, 1)  # Context
        data_frame_input = test_functions.MLI2TC1(conv_array, data_frame_input, 2)  # Context

    if is_MLI3TC1:
        # Check responses to see how likely they are to be coherent ones w.r.t the input.
        data_frame = test_functions.MLI3TC1(conv_array, data_frame, 1)  # Last answer
        data_frame_input = test_functions.MLI3TC1(conv_array, data_frame_input, 2)  # Last answer

    if is_analyze_question_freq:
        # Check for recurring questions and add metric to dataframe
        test_functions.analyze_question_freq(conv_chatter1, data_frame)
        test_functions.analyze_question_freq(conv_chatter2, data_frame_input)

    if is_MLA6TC1:
        # Check for stuttering using N-grams, and add metric to dataframe
        data_frame = test_functions.MLA6TC1(conv_chatter1, data_frame)
        data_frame_input = test_functions.MLA6TC1(conv_chatter2, data_frame_input)

    if not is_load_conversation:
        data_frame = test_functions.analyze_times(data_frame, chatter1_times)
        data_frame_input = test_functions.analyze_times(data_frame_input, chatter2_times)

    global df_summary
    global df_input_summary

    df_summary = util_functions.add_column(data_frame, df_summary)
    df_input_summary = util_functions.add_column(data_frame_input, df_input_summary)
    return df_summary, df_input_summary


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
    # Data frames containing all the data frames collected from each conversation per chatter
    df_summary = None
    df_input_summary = None

    script_start_time = time.time()
    for run in range(max_runs):
        convarray.clear()

        print('Starting conversation ' + str(run + 1))
        start_time = time.time()
        chatter1_times = []
        chatter2_times = []

        if not is_load_conversation:
            # Load conversation
            print("Generating conversation...")
            convarray, chatter1_times, chatter2_times = generate_conversation()
        else:
            print("Loading conversation...")
            convarray = load_conversation()

        if is_analyze_conversation:
            # Starts the analysis of the conversation
            print("Analyzing conversation...")
            df_summary, df_input_summary = analyze_conversation(convarray, chatter1_times, chatter2_times)
            print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # The method for presenting the metrics into a .xlsx-file. Will print both the summary-Dataframes to .xlsx
    print("Exporting results...")
    write_to_excel(df_summary, save_analysis_names[0])
    write_to_excel(df_input_summary, save_analysis_names[1])

    print("Done!")
    print('Total time the script took was: ' + str(time.time() - script_start_time) + 's')