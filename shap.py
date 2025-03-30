# shap_explainer.py

import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sms-spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['transformed_text'] = df['message'].apply(transform_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['transformed_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (optional if not loaded)
model = LogisticRegression()
model.fit(X_train, y_train)

# SHAP Explanation
explainer = shap.LinearExplainer(model, X_train.toarray(), feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test.toarray())

# Summary plot (top 20 words)
shap.summary_plot(shap_values, X_test.toarray(), feature_names=cv.get_feature_names_out(), max_display=20)

# Optional: Force plot for a single prediction
shap.initjs()
index = 5  # change this to view another test sample
shap.force_plot(
    explainer.expected_value,
    shap_values[index],
    features=X_test.toarray()[index],
    feature_names=cv.get_feature_names_out(),
    matplotlib=True
)

# Save model and vectorizer if needed
# joblib.dump(model, "model.pkl")
# joblib.dump(cv, "vectorizer.pkl")1