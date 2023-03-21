import inflect
import pandas as pd

p = inflect.engine()

df = pd.read_csv(filepath_or_buffer="ticket_generation/data/national_M2021_dl.csv")

# Transform all occupations from plural to singular
# In original dataset all occupations are in plural form ( ex. Sales Managers )
df["OCC_TITLE"] = df["OCC_TITLE"].apply(p.singular_noun)

# Remove rows in which conversion has given errors
df = df[df["OCC_TITLE"] != False]

df["OCC_TITLE"] = df["OCC_TITLE"].apply(str.lower)

df.to_csv(path_or_buf="ticket_generation/data/national_M2021_dl_processed.csv", index=False)
