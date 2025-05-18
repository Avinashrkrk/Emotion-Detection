import pandas as pd
import joblib
import re
import neattext.functions as nfx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(filepath):
    texts, emotions = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    texts.append(parts[0])
                    emotions.append(parts[1])
    return pd.DataFrame({'Text': texts, 'Emotion': emotions})

# Loading train and test data set
train_df = load_data('./data/train.txt')
test_df = load_data('./data/test.txt')

# Text cleaning using neattext
def clean_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    return text.lower()

train_df['Clean_Text'] = train_df['Text'].apply(clean_text)
test_df['Clean_Text'] = test_df['Text'].apply(clean_text)

# Encoding emotions
le = LabelEncoder()
train_df['Label'] = le.fit_transform(train_df['Emotion'])
test_df['Label'] = le.transform(test_df['Emotion'])

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), # To handle unigrams or biggrams
    max_features=6000,
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df['Clean_Text'])
X_test = vectorizer.transform(test_df['Clean_Text'])

y_train = train_df['Label']
y_test = test_df['Label']

# Model Training
model = LogisticRegression(
    C=1.5,
    class_weight='balanced',
    max_iter=300,
    solver='liblinear'
)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Saving a model for prediction without retraining
joblib.dump(model, 'emotion_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')
