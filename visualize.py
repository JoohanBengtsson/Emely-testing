import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Test case for REQ_P2: Assessing the toxicity-levels of any text input, a text-array of any size
def viz_REQ_P2(path):
    print("Time to plot " + path)

    df = pd.read_csv(path)

    print(df)

    sns.displot(df, x="toxicity", kde=True)
    plt.show()

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
    merged_data.to_csv(path + "/tmp.csv")
    print(merged_data["toxicity"].describe(include="all"))

    toxic_plot = sns.distplot(merged_data, kde=True, hist=True, hist_kws={"range": [0, 1]})
    toxic_plot.set(xlabel="Toxicity", ylabel="Frequency")
    plt.show()

#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v02")
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v03")
#merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Emely_v04")
merge_and_plot_REQ_P2("E:/SynologyDrive/research/_Emely/REQ_P2-toxicity/Blenderbot")
