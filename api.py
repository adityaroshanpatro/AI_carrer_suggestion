from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("career_path_encoder.pkl", "rb") as enc_file:
    career_path_encoder = pickle.load(enc_file)

with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    vectorizer = pickle.load(tfidf_file)

@app.route("/")
def home():
    return "Welcome to the Career Path Prediction API! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if "candidate_skills" not in data or "candidate_level" not in data:
        return jsonify({"error": "Missing 'candidate_skills' or 'candidate_level' in request body"}), 400

    candidate_skills = data["candidate_skills"]

    # Convert skills to TF-IDF vector
    skills_tfidf = vectorizer.transform([candidate_skills]).toarray()
    skills_df = pd.DataFrame(skills_tfidf, columns=vectorizer.get_feature_names_out())

    # Make prediction
    prediction = model.predict(skills_df)[0]
    predicted_career = career_path_encoder.inverse_transform([prediction])[0]

    return jsonify({"predicted_career_path": predicted_career})

if __name__ == "__main__":
    app.run(debug=True)
