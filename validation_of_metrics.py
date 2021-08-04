# from util_functions import
import random

import pandas as pd

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


def assess_MLI3TC1():
    with open('validation/saved_conversation.txt') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    nsp_assessment = []
    input_sentences = []
    response_sentences = []
    for j in range(2):
        indices_list = list(range(len(lines)))
        random.shuffle(indices_list)
        for i in range(1, len(lines), 2):
            prev_row = lines[indices_list.pop(0)]
            row = lines[indices_list.pop(0)]
            row = row.split('\n')[0]
            prev_row = prev_row.split('\n')[0]
            nsp_points = util_functions.nsp(prev_row, row)
            nsp_assessment.append(nsp_points[0] - nsp_points[1])
            input_sentences.append(prev_row)
            response_sentences.append(row)
        inputs = [input_sentences, response_sentences, nsp_assessment]
        df2 = pd.DataFrame(columns=['Input sentences', 'Response sentences', 'NSP assessments'], data=inputs)
        df = df.append(response_sentences)
        df = df.append(nsp_assessment)
        #df.insert(0, 'NSP assessment', nsp_assessment)
        #df.insert(0, 'Response sentences', response_sentences)
        #df.insert(0, 'Input sentences', input_sentences)
    df = df.sort_values(by=['NSP assessment'])
    f.close()
    return df


if __name__ == "__main__":
    if test_MLI3TC1:
        df = assess_MLI3TC1()

    if is_stutter:
        pass
        # check_stutter(textarray, df_output)

    # df_output.to_excel("./reports/validation_of_metrics.xlsx")
    df.to_excel("./reports/validation_of_metrics.xlsx")
