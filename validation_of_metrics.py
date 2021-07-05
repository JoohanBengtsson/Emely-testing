from main import check_stutter
import pandas as pd

# Determine which metrics should be tested
is_stutter = True

# Read the text file that is tested.
textfile = open("validation_text.txt",'r')
text = textfile.read()
text = text.replace('!','.')
text = text.replace('?','.')
text = text.replace(':','.')
text = text.replace('...','.')
textarray = text.split('.')

# The output dataframe
df_output = pd.DataFrame(data=textarray)
if __name__ == "__main__":
    if is_stutter:
        check_stutter(textarray, df_output)

    df_output.to_excel("./reports/validation_of_metrics.xlsx")

