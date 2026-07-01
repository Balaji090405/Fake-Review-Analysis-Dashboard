import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df['review']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

df['fake_probability'] = model.predict_proba(X_vec)[:,1]

df.to_csv("output.csv", index=False)

print("Fake review model completed!")