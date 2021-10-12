# from util_functions import
import random

import pandas as pd
import numpy
import sys
from os import path

sys.path.append(path.abspath("BERT-SQuAD"))
from bert import QA

# Determine which metrics should be tested
import util_functions

is_stutter = True
test_TC_REQ_I3 = False
test_QA = True


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

def assessment_method(load_conv_folder, test_case):
    print("Loading conversation...")
    lines, test_ids, test_sets = util_functions.load_conversation(load_conv_folder, 0)
    input_sents = []
    response_sents = []
    for i in range(0, len(lines) - 1, 2):
        input_sents.append(lines[i])
        response_sents.append(lines[i + 1])
    df = pd.DataFrame()
    used_sent_pairs = {}
    for j in range(16):
        print("permutation {perm}".format(perm=j))
        indices_list_prev = list(range(len(lines)))
        indices_list = list(range(len(response_sents)))
        random.shuffle(indices_list_prev)
        random.shuffle(indices_list)

        model_assessments = []
        input_sentences = []
        questions = []
        response_sentences = []

        if test_case == 'QA':
            test_set_keys = test_sets.keys()
            for key in test_set_keys:
                test_set_list = test_sets[key]
                for test_set in test_set_list:
                    # Extract the answers only given after the question. Sometimes the real answer, sometimes a
                    # randomized answer.
                    disturbance_index = int(5 * random.random() - 10)
                    answers, test_idx = util_functions.extract_answers([response_sents[i + disturbance_index] for i in
                                                                        indices_list], [test_ids[i] for i in
                                                                        indices_list], 1000000 + int(key[3]) * 10000 + test_set["id"] + 0.5)
                    #questions.append([input_sents[i] for i in indices_list])

                    if len(answers) > 0:
                        if not test_set["directed"]:
                            # Reduce the answer to the specific answer to the question.
                            interpret = util_functions.openQA(answers, test_set["QA"])
                            model_assessments.append(
                                util_functions.check_similarity([test_set["answer"]] * len(interpret), interpret))
                            input_sentences.append(interpret)
                            response_sentences.append([test_set["answer"]] * len(interpret))

                input_matrix = [input_sentences[0], response_sentences[0], model_assessments[0]]
                input_matrix = numpy.transpose(input_matrix)

                df2 = pd.DataFrame(columns=['Interpretation', 'Answer', test_case + ' assessments'],
                                   data=input_matrix)
                df = df.append(df2)

        if test_case != 'QA':
            for i in range(len(input_sents)):
                prev_index = indices_list_prev.pop(0)
                index = response_sents[indices_list.pop(0)]
                combined_sents = lines[prev_index] + ':' + lines[index]
                if combined_sents not in used_sent_pairs:
                    if test_case == 'TC_REQ_I3':
                        model_points = util_functions.nsp(lines[prev_index], lines[index])
                        model_assessments.append(model_points[0] - model_points[1])
                    input_sentences.append(lines[prev_index])
                    response_sentences.append(lines[index])
                    used_sent_pairs[combined_sents] = 1
            input_matrix = [input_sentences, response_sentences, model_assessments]
            input_matrix = numpy.transpose(input_matrix)

            df2 = pd.DataFrame(columns=['Prompt sentences', 'Reply sentences', test_case + ' assessments'],
                               data=input_matrix)
            df = df.append(df2)
            del df2
    df = df.sort_values(by=test_case + ' assessments', ascending=False)
    df.to_excel("./reports/validation/validation_" + test_case + ".xlsx")


if __name__ == "__main__":
    if test_TC_REQ_I3:
        assessment_method("validation_NSP/", 'TC_REQ_I3')

    if test_QA:
        assessment_method("validation_QA/", 'QA')

    if is_stutter:
        pass
        # check_stutter(textarray, df_output)

    # df_output.to_excel("./reports/validation_of_metrics.xlsx")
    # df.to_excel("./reports/validation_of_metrics.xlsx")
