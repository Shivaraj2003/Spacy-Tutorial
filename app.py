import os
import whisper
import spacy
import pandas as pd
import re
from flask import Flask, request, jsonify

# Load the symptoms data
symptom_data = pd.DataFrame({
    "Symptom": [
        "fever", "chills", "fatigue", "malaise", "weakness",
        "weight loss", "weight gain", "headache", "abdominal pain",
        "chest pain", "joint pain", "muscle pain", "back pain",
        "cough", "shortness of breath", "difficulty breathing",
        "wheezing", "sore throat", "nausea", "vomiting", "diarrhea",
        "constipation", "bloating", "heartburn", "dizziness",
        "lightheadedness", "tingling", "numbness", "seizures",
        "tremors", "palpitations", "irregular heartbeat",
        "chest tightness", "fainting", "rash", "itching", "redness",
        "swelling", "blisters", "bruising", "frequent urination",
        "painful urination", "blood in urine", "urinary incontinence",
        "anxiety", "depression", "mood swings", "confusion",
        "forgetfulness", "blurred vision", "red eyes", "itchy eyes",
        "dry eyes", "eye pain"
    ],
    "Diagnosis": [
        # Add the corresponding diagnosis
    ],
    "Treatment": [
        # Add the corresponding treatment
    ]
})

# Step 1: Transcribe audio to text
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Step 2: Extract names, ages, and symptoms
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    names, ages = [], []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)

    name_matches = re.findall(r"(?:I['’]m|My name is|This is)\s+([A-Z][a-z]+)", text)
    names.extend(name_matches)

    age_matches = re.findall(r"(\b\d{1,3}\b)\s+(years old|year old|aged)", text, re.IGNORECASE)
    extracted_ages = [match[0] for match in age_matches]

    symptom_keywords = symptom_data["Symptom"].str.lower().tolist()
    symptom_patterns = r"|".join(symptom_keywords)
    symptom_matches = re.findall(symptom_patterns, text, re.IGNORECASE)

    matched_entities = []
    for name in names:
        name_position = text.find(name)
        closest_age, closest_distance = None, float('inf')
        
        for match in age_matches:
            age_position = text.find(match[0])
            distance = abs(name_position - age_position)
            if distance < closest_distance:
                closest_age = match[0]
                closest_distance = distance

        unique_symptoms = set(symptom_matches)
        symptoms_data = []
        for symptom in unique_symptoms:
            row = symptom_data[symptom_data["Symptom"].str.lower() == symptom.lower()]
            if not row.empty:
                diagnosis = row["Diagnosis"].iloc[0]
                treatment = row["Treatment"].iloc[0]
                symptoms_data.append({
                    "Symptom": symptom,
                    "Diagnosis": diagnosis,
                    "Treatment": treatment
                })

        symptom_str = ", ".join([data["Symptom"] for data in symptoms_data])
        diagnosis_str = "; ".join([data["Diagnosis"] for data in symptoms_data])
        treatment_str = "; ".join([data["Treatment"] for data in symptoms_data])

        matched_entities.append({
            "Name": name,
            "Age": closest_age if closest_age else "",
            "Symptoms": symptom_str,
            "Diagnosis": diagnosis_str if diagnosis_str else "Unknown",
            "Treatment": treatment_str if treatment_str else "No treatment available",
            "AudioFile": ""  # Placeholder for audio file
        })

    return matched_entities

# Flask server
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"  # Define a directory for saving files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    audio_file = request.files['file']
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    try:
        transcript = transcribe_audio(audio_path)
        entities = extract_entities(transcript)
        os.remove(audio_path)  # Clean up the file after processing
        return jsonify(entities)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
