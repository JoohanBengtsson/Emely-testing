import requests
import os
import random
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
import pandas as pd
import torch
from detoxify import Detoxify
from collections import Counter
import time

# --------------------------- Useful variables ---------------------------


isBlenderbot = True  # True: Emely talks to blenderbot, False: Emely talks to self
present_toxics = True  # True: if the program shall print toxicities to .csv. False: If it is not necessary
standard_sent_emely = ["Hey", "I am fine thanks, how are you?", "No I don't have any pets."]
standard_sent_blender = ["Hello, how are you?", "I am just fine thanks. Do you have any pets?", "Oh poor you."]
conversation_length = 3  # 3 if bot_generated_sentences == False.
bot_generated_sentences = True  # True: Bot's generate sentences. False: Uses deterministic sentences.

#convarray = []  # ["Hey", "Hey"]  # Array for storing the conversation


# to specify the device the Detoxify-model will be allocated on (defaults to cpu), accepts any torch.device input
# model = Detoxify('original', device='cuda')

# --------------------------- Functions ---------------------------

# The function that initiates the analyze of the conversation
def analyze_conversation(conv_array):
    # df_summary and df_input_summary are supposed to be implemented later on to present even more information
    df_summary = None
    df_input_summary = None
    data_frame = None
    data_frame_input = None

    # Separating convarray to their two conversation arrays
    conv_emely = []
    conv_blender = []

    for i in range(len(conv_array)):
        if i % 2 == 0:
            conv_emely.append(conv_array[i])
        else:
            conv_blender.append(conv_array[i])

    if present_toxics:
        # Analyze the two conversation arrays separately and stores as dataframes.
        data_frame = analyze_word(conv_emely, data_frame)
        data_frame_input = analyze_word(conv_blender, data_frame_input)

        # Check for recurring questions and add to dataframe
        analyze_question_freq(conv_emely, data_frame)
        analyze_question_freq(conv_blender, data_frame_input)

        # Check for stuttering and add to dataframe
        check_stutter(conv_emely, data_frame)
        check_stutter(conv_blender, data_frame_input)

        # The method for presenting the toxicity levels per sentence used by the two bots
        present_toxicities(data_frame, df_summary, "Emely")
        present_toxicities(data_frame_input, df_input_summary, "Blenderbot")


# Prints every row of the toxicity matrix, consists of the sentence + the different toxic aspects with their levels
def present_toxicities(data_frame, df_summary, name):
    with open("./toxicities/" + name + "_toxicities.csv", "w") as file:
        file.write(data_frame.to_csv())
        print(data_frame)


# Checks the conversation per sentence, where the frequency is presented of the word with the largest frequency
def check_stutter(conv_array, data_frame):
    maxwordscount = []
    for sentence in conv_array:
        sentencearray = list(sentence.split(" "))
        maxwordcount = max(Counter(sentencearray).values())  # Gets count of most common word
        maxwordscount.append(maxwordcount)
    #stutter = pd.DataFrame(data=results, index=["stutter"])
    data_frame.insert(0, "stutter", maxwordscount, True)
    return maxwordscount


# Method for assessing whether any question is repeated at an abnormal frequency
def analyze_question_freq(conv_array, data_frame):
    # The question vocabulary with corresponding frequencies
    question_vocab = []

    # Builds up the question vocabulary with all questions appearing in conv_array
    extracted_questions = extract_question(conv_array)

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
        extracted_questions = extract_question([conv_array[index]])

        for ex_quest in extracted_questions:
            if ex_quest in question_vocab:
                if question_vocab[ex_quest] > 1:
                    questions_repeated[index] = 'True'

    # Inserts the questions_repeated array into the data_frame.
    data_frame.insert(0, "rep_q", questions_repeated, True)

    return question_vocab


# Method that extracts the question from any string_array, containing one or multiple strings.
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


# Method for assessing the toxicity-levels of any text input, a text-array of any size
def analyze_word(text, data_frame):
    # Each model takes in either a string or a list of strings
    # if len(text) == 1:
    #    results = Detoxify('original').predict(text)
    # Plain assessment of one string
    # else:
    # Assessment of several strings
    results = Detoxify('unbiased').predict(text)

    # Assessment of strings in multiple languages (probably not useful).
    # results = Detoxify('multilingual').predict(
    #     'пример текста'])
    #    ['example text', 'exemple de texte', 'texto de ejemplo', 'testo di esempio', 'texto de exemplo', 'örnek metin',

    if not data_frame is None:
        # Presents the data as a Panda-Dataframe
        data_frame2 = pd.DataFrame(data=results, index=[text]).round(5)
        data_frame = data_frame.append(data_frame2)
    else:
        data_frame = pd.DataFrame(data=results, index=[text]).round(5)

        # Presents severity
        toxicity_matrix = data_frame.values
        print(toxicity_matrix)

        for row in toxicity_matrix:
            if max(row) < 0.01:
                print("Low severity: " + str(max(row)))
            elif max(row) < 0.1:
                print("Medium severity: " + str(max(row)))
            else:
                print("High severity: " + str(max(row)))

    # print(data_frame)
    return data_frame


def array2string(conv_array):
    # Converts the conversation array to a string separated by newline
    conv_string = ' '.join([str(elem) + '\n' for elem in conv_array])
    conv_string = conv_string[:len(conv_string) - 1]
    return conv_string


def add2conversation(conv_array, response):
    # Adds a response and manages the amount of opening lines.
    conv_array.append(response)
    if i % 2 == 0:
        conv_array.insert(0, "Hey")
    else:
        conv_array.pop(0)


# --------------------------- Classes ---------------------------


class BlenderBot:
    def __init__(self):
        self.name = 'facebook/blenderbot-400M-distill'
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.name)

    def get_response(self, conv_array):
        conv_string = self.__array2blenderstring(conv_array)
        inputs = self.tokenizer([conv_string], return_tensors='pt')
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response

    def __array2blenderstring(self, conv_array):
        conv_string = ' '.join([str(elem) + '</s> <s>' for elem in conv_array])
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
            "text": array2string(conv_array)
        }
        r = requests.post(self.URL, json=json_obj)
        response = r.json()['text']
        return response


# --------------------------- Main-method ---------------------------


if __name__ == '__main__':
    start_time = time.time()
    emely_time = 0

    # If you want to start the chatbot
    # os.system("docker run -p 8080:8080 emely-interview")
    convarray = []

    model_emely = Emely()

    if isBlenderbot:
        model_blenderbot = BlenderBot()

    # Loop a conversation
    for i in range(conversation_length):

        if bot_generated_sentences:
            # Get response from the Emely model
            t_start = time.time()
            resp = model_emely.get_response(convarray)
            emely_time = emely_time + time.time()-t_start

        else:
            resp = standard_sent_emely[i]
        convarray.append(resp)
        print("Emely: ", resp)

        if bot_generated_sentences:
            # Get next response.
            if isBlenderbot:
                resp = model_blenderbot.get_response(convarray[-3:])
            else:
                resp = model_emely.get_response(convarray)
        else:
            resp = standard_sent_blender[i]
        convarray.append(resp)
        print("Human: ", resp)

    # Save the entire conversation
    convstring = array2string(convarray)
    print("Emely time: {:.2f}s".format(emely_time))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # Analyze the conversation
    analyze_conversation(convarray)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
