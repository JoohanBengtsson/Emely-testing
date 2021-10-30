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

# Plot distributions for stuttering
def viz_REQ_I3(path):
    nbr_dialogs = 0
    dialogs_with_incoherence = 0
    incohence_per_dialog = []

    for filename in os.listdir(path):
        if ("I3_runID" in filename):
            start_new_file = True
            nbr_dialogs += 1
            full_path = path + "/" + filename
            file = open(full_path)
            csvreader = csv.reader(file)
            incoherence_counter = 0
            for row in csvreader:
                incoherence_counter += 1
                if (start_new_file):
                    dialogs_with_incoerence += 1
                    start_new_file = False
            incohence_counincohence_per_dialog.append(incoherence_counter)
        else:
            incohence_per_dialog.append(0)

    print("Dialogs with stuttering present: " + str(dialogs_with_incoherence) + " out of " + str(nbr_dialogs))

    output = pd.DataFrame(incohence_per_dialog)

    print("Incoherence statistics: ")
    print(output.describe())

    incoherence_plot = sns.distplot(output, kde=True, hist=True)
    incoherence_plot.set(xlabel="Incoherence presence", ylabel="Frequency")
    plt.show()

# REQ I3

# REQ A4
viz_REQ_A4("C:/BorgCloud/research/_Emely/REQ_A4_stuttering/Emely_v02")

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
