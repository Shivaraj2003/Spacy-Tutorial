import os
import whisper
import spacy
import pandas as pd
import re
from flask import Flask, request, jsonify

# Load the symptoms data from a CSV file (assuming it's in the same directory)
symptom_data = pd.read_csv("symptoms.csv")

# Step 1: Transcribe audio to text
def transcribe_audio(audio_bytes):
    """
    Transcribes audio data in bytes format using Whisper.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_bytes)
    return result["text"]

# Step 2: Extract names, ages, and symptoms
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    names = []
    ages = []

    # Extract names using spaCy
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)

    # Supplementary name extraction using regex
    name_matches = re.findall(r"(?:I['’]m|My name is|This is)\s+([A-Z][a-z]+)", text)
    names.extend(name_matches)

    # Extract ages using regex
    age_matches = re.findall(r"(\b\d{1,3}\b)\s+(years old|year old|aged)", text, re.IGNORECASE)
    extracted_ages = [match[0] for match in age_matches]

    # Extract symptoms
    symptom_keywords = symptom_data["Symptom"].str.lower().tolist()
    symptom_patterns = r"|".join(symptom_keywords)
    symptom_matches = re.findall(symptom_patterns, text, re.IGNORECASE)

    matched_entities = []
    for name in names:
        name_position = text.find(name)

        # Find closest age to the person's name
        closest_age = None
        closest_distance = float('inf')
        for match in age_matches:
            age_position = text.find(match[0])
            distance = abs(name_position - age_position)
            if distance < closest_distance:
                closest_age = match[0]
                closest_distance = distance

        # Retrieve diagnosis and treatment for matched symptoms
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
            "Treatment": treatment_str if treatment_str else "No treatment available"
        })

    return matched_entities

# Flask server for processing audio
app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    audio_file = request.files['file']

    # Read the audio data into bytes
    audio_bytes = audio_file.read()

    # Transcribe and extract
    transcript = transcribe_audio(audio_bytes)
    entities = extract_entities(transcript)

    # **No file storage on Render.com (optional):**
    # Since temporary file storage is limited on Render.com, you
    # can directly process the audio bytes without saving them
    # to disk. However, if you need to store the audio file for
    # later processing or analysis, you might consider using a
    # cloud storage service like Google Cloud Storage or AWS S3.

    return jsonify(entities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)