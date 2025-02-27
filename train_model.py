import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
job_postings = pd.read_csv("job_postings.csv")
job_skills = pd.read_csv("job_skills.csv")
job_summary = pd.read_csv("job_summary.csv")

# Data Cleaning
job_postings.drop_duplicates(inplace=True)
job_skills.drop_duplicates(inplace=True)
job_summary.drop_duplicates(inplace=True)
job_skills["job_skills"].fillna("Unknown", inplace=True)
job_postings["job_location"].fillna("Unknown", inplace=True)

# Normalize text columns
for col in ["job_title", "company"]:
    job_postings[col] = job_postings[col].str.lower().str.strip()
job_skills["job_skills"] = job_skills["job_skills"].str.lower().str.strip()
job_summary["job_summary"] = job_summary["job_summary"].str.lower().str.strip()

# Merge data
df = job_postings.merge(job_skills, on="job_link", how="inner").merge(job_summary, on="job_link", how="inner")

# Categorize job titles
def categorize_job_title(title):
    title = title.lower()
    categories = {
        "data science & ai": ["data scientist", "machine learning", "ai"],
        "software development": ["software engineer", "developer", "backend", "frontend"],
        "data & business analytics": ["business analyst", "data analyst", "analytics"],
        "cloud & devops": ["cloud", "devops"],
        "cybersecurity": ["cybersecurity", "security"],
        "product & project management": ["product manager", "project manager"]
    }
    for category, keywords in categories.items():
        if any(kw in title for kw in keywords):
            return category
    return "other"

df["career_path"] = df["job_title"].apply(categorize_job_title)

# Encode categorical features
career_path_encoder = LabelEncoder()
df["career_path_encoded"] = career_path_encoder.fit_transform(df["career_path"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
skills_tfidf = vectorizer.fit_transform(df["job_skills"])
skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

df_model = pd.concat([df, skills_df], axis=1)
df_model.drop(columns=["job_link", "job_skills", "career_path", "job_level", "last_processed_time", "last_status", "got_summary", "got_ner"], errors="ignore", inplace=True)

X = df_model.drop(columns=["career_path_encoded"], errors="ignore")
y = df_model["career_path_encoded"]
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
X = X.drop(columns=non_numeric_cols, errors="ignore")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("career_path_encoder.pkl", "wb") as enc_file:
    pickle.dump(career_path_encoder, enc_file)

with open("tfidf_vectorizer.pkl", "wb") as tfidf_file:
    pickle.dump(vectorizer, tfidf_file)

print("🎉 Model training complete! Saved as model.pkl")
