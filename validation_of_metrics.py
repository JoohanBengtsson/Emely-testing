# from util_functions import
import random

import pandas as pd
import numpy

# Determine which metrics should be tested
import util_functions

is_stutter = True
test_MLI3TC1 = True


# Read the text file that is tested.
# textfile = open("validation_text.txt", 'r')
# text = textfile.read()
# text = text.replace('!', '.')
# text = text.replace('?', '.')
# text = text.replace(':', '.')
# text = text.replace('...', '.')
# textarray = text.split('.')

# The output dataframe
# df_output = pd.DataFrame(data=textarray)


def assessment_method(test_case):
    with open('validation/saved_conversation.txt') as f:
        lines = f.readlines()
    input_sents = []
    response_sents = []
    for i in range(0, len(lines), 2):
        input_sents.append(lines[i])
        response_sents.append(lines[i + 1])
    df = pd.DataFrame()
    used_sent_pairs = {}
    for j in range(16):
        indices_list_prev = list(range(len(lines)))
        indices_list = list(range(len(response_sents)))
        random.shuffle(indices_list_prev)
        random.shuffle(indices_list)

        model_assessments = []
        input_sentences = []
        response_sentences = []
        for i in range(len(input_sents)):
            prev_row = lines[indices_list_prev.pop(0)]
            row = response_sents[indices_list.pop(0)]
            row = row.split('\n')[0]
            prev_row = prev_row.split('\n')[0]
            combined_sents = prev_row + ':' + row
            if combined_sents not in used_sent_pairs:
                if test_case == 'MLI3TC1':
                    model_points = util_functions.nsp(prev_row, row)
                    model_assessments.append(model_points[0] - model_points[1])
                input_sentences.append(prev_row)
                response_sentences.append(row)
                used_sent_pairs[combined_sents] = 1
        input_matrix = [input_sentences, response_sentences, model_assessments]
        input_matrix = numpy.transpose(input_matrix)

        df2 = pd.DataFrame(columns=['Input sentences', 'Response sentences', test_case + ' assessments'],
                           data=input_matrix)
        df = df.append(df2)
        del df2
    df = df.sort_values(by='NSP assessments', ascending=False)
    f.close()
    df.to_excel("./reports/validation_" + test_case + ".xlsx")


if __name__ == "__main__":
    if test_MLI3TC1:
        assessment_method('MLI3TC1')

    if is_stutter:
        pass
        # check_stutter(textarray, df_output)

    # df_output.to_excel("./reports/validation_of_metrics.xlsx")
    # df.to_excel("./reports/validation_of_metrics.xlsx")
