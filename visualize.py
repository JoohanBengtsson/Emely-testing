import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Merge csv-files for REQ_P2
def merge_and_plot_REQ_P2(path):
    header = []
    rows = []

    for filename in os.listdir(path):
        start_new_file = True
        if ("v2" in filename):
            full_path = path + "/" + filename

            file = open(full_path)
            csvreader = csv.reader(file)

            if (len(header) == 0): # only needed once
                header = next(csvreader)
            else:
                next(csvreader) # drop the header if it has already been added

            for row in csvreader:
                num_values = list(map(float, row))
                rows.append(num_values)

            #print(rows)

            # with open(full_path, newline='') as csvfile:
            #     reader = csv.reader(full_path, delimiter=',')
            #     for row in reader:
            #         if (not first_row_exists):
            #             print(str(row))
            #             #merged_content += row
            #             print("row my boat")

    # put all content in a data frame
    merged_data = pd.DataFrame(rows, columns=header)
    merged_data.to_csv(path + "/tmp_REQ_P2.csv")
    nbr_toxic_replies = merged_data[merged_data >= 0.1].count()
    print("Number of toxic replies:\n" + str(nbr_toxic_replies) + " out of " + str(len(rows)))
    print(merged_data["toxicity"].describe(include="all"))

    toxic_plot = sns.distplot(merged_data, kde=True, hist=True, hist_kws={"range": [0, 1]})
    toxic_plot.set(xlabel="Toxicity", ylabel="Frequency")
    plt.show()

# Test case for REQ_A3: Question nagging
def viz_REQ_A3(path):
    nbr_dialogs = 0
    dialogs_with_nagging_questions = 0
    distribution_of_repeated_questions = [0]*25
    nagging_questions = []

    for filename in os.listdir(path):
        if ("A3_runID" in filename):

            start_new_file = True
            nbr_dialogs += 1

            full_path = path + "/" + filename

            file = open(full_path)
            csvreader = csv.reader(file)

            for row in csvreader:
                if (int(row[1]) > 1):
                    if (start_new_file):
                        dialogs_with_nagging_questions += 1
                        start_new_file = False
                    distribution_of_repeated_questions[(int(row[1]))] += 1
                    nagging_questions.append(row)

    print("Dialogs with nagging questions: " + str(dialogs_with_nagging_questions) + " out of " + str(nbr_dialogs))
    print("Distribution: ")
    print(distribution_of_repeated_questions)

    output = pd.DataFrame(nagging_questions)
    output.to_csv(path + "/REQ_A3_examples.csv")

    print(output.describe())
    print(output.median())

    plt.bar(range(25), distribution_of_repeated_questions)
    plt.show()

# Plot distributions for stuttering
def viz_REQ_A4(path):
    nbr_dialogs = 0
    dialogs_with_stuttering = 0
    stuttering_scores = []
    most_stuttered_example = ""
    highest_stuttering_score = 0

    for filename in os.listdir(path):
        if ("A4_runID" in filename):
            start_new_file = True
            nbr_dialogs += 1
            full_path = path + "/" + filename
            file = open(full_path)
            csvreader = csv.reader(file)
            for row in csvreader:
                current_stuttering_score = float(row[1])
                if (current_stuttering_score > 0):
                    if (start_new_file):
                        dialogs_with_stuttering += 1
                        start_new_file = False
                    stuttering_scores.append(float(row[1]))
                    if (current_stuttering_score > highest_stuttering_score):
                        most_stuttered_example = row[0]
                        highest_stuttering_score = current_stuttering_score

    print("Dialogs with stuttering present: " + str(dialogs_with_stuttering) + " out of " + str(nbr_dialogs))
    print("Most stuttered example: " + most_stuttered_example + " (Score: " + str(highest_stuttering_score) + ")")

    output = pd.DataFrame(stuttering_scores)

    print("Stuttering statistics: ")
    print(output.describe())

    stuttering_plot = sns.distplot(output, kde=True, hist=True)
    stuttering_plot.set(xlabel="Stuttering scores", ylabel="Frequency")
    plt.show()

# Plot distributions for dialog coherence
def viz_REQ_I2(path, nbr_dialogs):
    dialogs_with_incoherence = 0
    incoherence_per_dialog = []

    for filename in os.listdir(path):
        if ("I3_runID" in filename):
            start_new_file = True
            full_path = path + "/" + filename
            file = open(full_path)
            csvreader = csv.reader(file)
            incoherence_counter = 0
            for row in csvreader:
                incoherence_counter += 1
                if (start_new_file):
                    dialogs_with_incoherence += 1
                    start_new_file = False
            incoherence_per_dialog.append(incoherence_counter)

    print("Dialogs with incoherence present: " + str(dialogs_with_incoherence) + " out of " + str(nbr_dialogs))

    output = pd.DataFrame(incoherence_per_dialog)

    print("Incoherence statistics: ")
    print(output.describe())

    incoherence_plot = sns.distplot(output, kde=False, hist=True)
    incoherence_plot.set(xlabel="Incoherence presence", ylabel="Frequency")
    plt.show()

# Test case for REQ_I1: Consistent self-image
def viz_REQ_I1(path, nbr_dialogs):
    dialogs_with_injected_test_cases = 0
    dialogs_with_failed_test_cases = 0
    injected_test_cases_per_dialog = [0] * nbr_dialogs
    passed_test_cases_per_dialog = [0] * nbr_dialogs
    failed_test_cases_per_dialog = [0] * nbr_dialogs

    distribution_of_failed_test_cases = []

    index = 0
    for filename in os.listdir(path):
        if ("I1_runID" in filename):
            start_new_file = True

            full_path = path + "/" + filename

            file = open(full_path)
            lines = file.readlines()

            for line in lines:
                expected = ""
                observed = ""
                verdict = ""
                if (line[0] == "E"): # Expected result found
                    expected = line
                elif (line[0] == "O"): # Observed result found
                    observed = line
                elif (line[0] == "T"): # Test verdict found
                    verdict = line
                    passed_test_cases = line.count("Pass")
                    passed_test_cases -= 1 # Since the first test cases shall not be counted - it cannot fail
                    failed_test_cases = line.count("Fail")
                    nbr_of_injected_test_cases = passed_test_cases + failed_test_cases
                    if (nbr_of_injected_test_cases > 0):
                        dialogs_with_injected_test_cases += 1
                        distribution_of_failed_test_cases.append(failed_test_cases)
                    if (failed_test_cases > 0):
                        dialogs_with_failed_test_cases += 1

                    injected_test_cases_per_dialog[index] = nbr_of_injected_test_cases
                    passed_test_cases_per_dialog[index] = passed_test_cases
                    failed_test_cases_per_dialog[index] = failed_test_cases

            index += 1

    print("Dialogs with failed test cases: " + str(dialogs_with_failed_test_cases) + " out of " + str(dialogs_with_injected_test_cases) + " dialogs with injected test cases.")
    #print("Distribution injected: ")
    #print(distribution_of_injected_test_cases)

    #print("Distribution passed: ")
    #print(distribution_of_passed_test_cases)

    #print("Distribution failed: ")
    #print(distribution_of_failed_test_cases)

    output = pd.DataFrame(distribution_of_failed_test_cases)
    output.to_csv(path + "/REQ_I1_results.csv")

    print(output.describe())
    print("Median: " + str(output[0].median()))

    consistency_plot = sns.distplot(output, kde=False, hist=True)
    consistency_plot.set(xlabel="Failed inconsistency test cases", ylabel="Frequency")
    plt.show()

# REQ I1
viz_REQ_I1("E:/SynologyDrive/research/_Emely/REQ_I1_consistency/Emely_v04", 200)

# REQ I3
#viz_REQ_I2("E:/SynologyDrive/research/_Emely/REQ_I2_dialog_coherence/Emely_v04", 200)

# REQ A4
#viz_REQ_A4("C:/BorgCloud/research/_Emely/REQ_A4_stuttering/Emely_v02")

# REQ A3
#viz_REQ_A3("E:/SynologyDrive/research/_Emely/REQ_A2_Nagging/Emely_v02")
#viz_REQ_A3("E:/SynologyDrive/research/_Emely/REQ_A2_Nagging/Emely_v03")
#viz_REQ_A3("E:/SynologyDrive/research/_Emely/REQ_A2_Nagging/Emely_v04")
#viz_REQ_A3("E:/SynologyDrive/research/_Emely/REQ_A2_Nagging/Blenderbot")

# REQ P2
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v02")
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v03")
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v04")
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Blenderbot")
