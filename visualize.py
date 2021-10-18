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

            for row in csvreader:
                rows.append(row)

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

    toxic_plot = sns.displot(merged_data, x="toxicity", kde=True)
    toxic_plot.set(xticklabels=[])
    plt.show()



merge_and_plot_REQ_P2("./reports/211017_223806_emely-8089_S1R200P50")
#viz_REQ_P2("REQ_P2_v3.csv")