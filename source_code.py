import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new input
sample = ["This is the best app I’ve used!", "I hate how it works."]
sample_vector = vectorizer.transform(sample)
predictions = model.predict(sample_vector)
print("Predictions:", predictions)
