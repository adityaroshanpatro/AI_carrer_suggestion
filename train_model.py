import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder

# Load data
job_postings = pd.read_csv("job_postings.csv")
job_skills = pd.read_csv("job_skills.csv")
job_summary = pd.read_csv("job_summary.csv")

# Data Cleaning
job_postings.drop_duplicates(inplace=True)
job_skills.drop_duplicates(inplace=True)
job_summary.drop_duplicates(inplace=True)

job_skills["job_skills"] = job_skills["job_skills"].fillna("Unknown")
job_postings["job_location"] = job_postings["job_location"].fillna("Unknown")

# Normalize text columns
for col in ["job_title", "company"]:
    job_postings[col] = job_postings[col].str.lower().str.strip()
job_skills["job_skills"] = job_skills["job_skills"].str.lower().str.strip()
job_summary["job_summary"] = job_summary["job_summary"].str.lower().str.strip()

# Merge datasets
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

# Vectorize skills using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
skills_tfidf = vectorizer.fit_transform(df["job_skills"])
skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Drop unnecessary columns
drop_columns = ["job_link", "job_skills", "career_path", "company", "job_level", 
                "last_processed_time", "last_status", "got_summary", "got_ner"]

df.drop(columns=drop_columns, errors="ignore", inplace=True)

# Merge TF-IDF features
df_model = pd.concat([df, skills_df], axis=1)

# Split data
X = df_model.drop(columns=["career_path_encoded"], errors="ignore")
y = df_model["career_path_encoded"]

# Save feature names for later validation
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
print("****************************")
# print(X_train.dtypes)  # Check for non-numeric columns
print("****************************")
label_encoders = {}

for col in X_train.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    X_train[col] = label_encoders[col].fit_transform(X_train[col])

    # Transform test set and handle unseen labels
    X_test[col] = X_test[col].apply(lambda x: label_encoders[col].transform([x])[0] 
                                    if x in label_encoders[col].classes_ 
                                    else -1)  # Assign -1 to unknown categories

print("Converted categorical columns to numeric format.")
print("****************************")
# Save feature names to ensure consistency during inference
with open("features.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)
print("Saved feature names for consistency.")
print("****************************")
model.fit(X_train, y_train)

# Save model & metadata
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Model training complete. Saved model.pkl & features.pkl")
