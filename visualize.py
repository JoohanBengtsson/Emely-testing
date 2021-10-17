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

viz_REQ_P2("testdata.csv")