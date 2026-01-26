import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("../data/SMSSpamCollection", sep="\t", names=["type", "message"])
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])
print(messages[0, :])
print(cv.get_feature_names_out()[886])