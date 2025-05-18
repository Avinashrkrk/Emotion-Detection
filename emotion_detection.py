import joblib
import neattext.functions as nfx

model = joblib.load('emotion_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def clean_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    return text.lower()

def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    label = model.predict(vectorized)[0]
    emotion = label_encoder.inverse_transform([label])[0]
    return emotion

if __name__ == '__main__':
    print("Emotion Detection - Type 'q' to quit")
    while True:
        inp = input("\nEnter a sentence: ")
        if inp.lower() == 'q':
            break
        result = predict_emotion(inp)
        print("Predicted Emotion:", result)