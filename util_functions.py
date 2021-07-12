# Adds dataset df_add to dataset df_original column-wise, regardless if df_original exists or not.
def add_column(df_add, df_original):
    if df_original is None:
        df_original = df_add
    else:
        df_original = df_original.append(df_add)
    return df_original

# Prints every row of the data_frame collecting all metrics. Writes to a Excel-file
def write_to_excel(df, name):
    df.to_excel("./reports/" + name + '_report.xlsx')

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
            return conv_array[(index+1):len(conv_array)]
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
