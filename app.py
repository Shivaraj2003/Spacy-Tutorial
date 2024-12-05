import os
import whisper
import tempfile
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import re
import spacy

# Load the symptom data
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
        "Infection", "Infection", "Chronic Fatigue Syndrome or Anemia",
        "Chronic Fatigue Syndrome or Anemia", "Chronic Fatigue Syndrome or Anemia",
        "Cancer or Hyperthyroidism", "Hypothyroidism or Obesity",
        "Migraine or Tension Headache", "Gastritis or Appendicitis",
        "Heart Attack or Angina", "Arthritis or Fibromyalgia",
        "Arthritis or Fibromyalgia", "Disc Herniation or Muscle Strain",
        "Bronchitis or Pneumonia", "Asthma or COPD", "Asthma or COPD",
        "Asthma or Allergic Reaction", "Pharyngitis or Tonsillitis",
        "Food Poisoning or Motion Sickness", "Food Poisoning or Motion Sickness",
        "Gastroenteritis or IBS", "IBS or Dehydration",
        "GERD or Lactose Intolerance", "GERD or Lactose Intolerance",
        "Vertigo or Hypotension", "Vertigo or Hypotension",
        "Neuropathy or Multiple Sclerosis", "Neuropathy or Multiple Sclerosis",
        "Epilepsy", "Parkinson’s Disease", "Arrhythmia or Anxiety",
        "Arrhythmia or Anxiety", "Heart Attack or Panic Attack",
        "Heart Attack or Panic Attack", "Allergic Reaction or Eczema",
        "Allergic Reaction or Eczema", "Allergic Reaction or Eczema",
        "Allergic Reaction or Eczema", "Herpes or Trauma", "Herpes or Trauma",
        "UTI or Diabetes Mellitus", "UTI or Bladder Stones",
        "Kidney Stones or Bladder Cancer", "Overactive Bladder or Weak Pelvic Floor Muscles",
        "Generalized Anxiety Disorder", "Generalized Anxiety Disorder or Bipolar Disorder",
        "Bipolar Disorder", "Dementia or Alzheimer’s Disease",
        "Dementia or Alzheimer’s Disease", "Conjunctivitis or Glaucoma",
        "Conjunctivitis or Glaucoma", "Allergies or Dry Eye Syndrome",
        "Allergies or Dry Eye Syndrome", "Uveitis or Eye Infection"
    ],
    "Treatment": [
        "Rest, hydration, and antipyretics", "Rest, hydration, and antipyretics",
        "Iron supplements, balanced diet, and regular exercise",
        "Iron supplements, balanced diet, and regular exercise",
        "Iron supplements, balanced diet, and regular exercise",
        "Medical evaluation and dietary supplements",
        "Thyroid hormone replacement and lifestyle changes",
        "Pain relievers, stress management, and sleep hygiene",
        "Antacids or surgery", "Emergency care and aspirin",
        "Pain relievers and physiotherapy", "Pain relievers and physiotherapy",
        "Rest and physiotherapy", "Cough suppressants and antibiotics",
        "Inhalers or oxygen therapy", "Inhalers or oxygen therapy",
        "Antihistamines or inhalers", "Saltwater gargle and antibiotics",
        "Antiemetics and rehydration", "Antiemetics and rehydration",
        "ORT and probiotics", "Fiber-rich diet and laxatives",
        "Antacids and dietary changes", "Antacids and dietary changes",
        "Rest and hydration", "Rest and hydration",
        "Vitamin B12 supplements", "Vitamin B12 supplements",
        "Antiepileptic drugs", "Levodopa and physiotherapy",
        "Beta-blockers and counseling", "Beta-blockers and counseling",
        "Emergency care and psychological support", "Emergency care and psychological support",
        "Antihistamines and corticosteroids", "Antihistamines and corticosteroids",
        "Antihistamines and corticosteroids", "Antihistamines and corticosteroids",
        "Antiviral drugs and wound care", "Antiviral drugs and wound care",
        "Antibiotics or blood sugar management",
        "Antibiotics or surgical intervention", "Pain management and evaluation",
        "Pelvic floor exercises and medications", "Counseling and antidepressants",
        "Counseling and antidepressants", "Counseling and antidepressants",
        "Cognitive therapy and medications", "Cognitive therapy and medications",
        "Eye drops or surgery", "Eye drops or surgery",
        "Artificial tears or antihistamines", "Artificial tears or antihistamines",
        "Anti-inflammatory or antibiotics"
    ]
})

# Initialize Flask app
app = Flask(__name__)

# Whisper Model Load
model = whisper.load_model("base")

# Step 1: Transcribe the audio
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# Step 2: Extract Symptoms and Provide Diagnosis/Treatment
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
            "Treatment": treatment_str if treatment_str else "No treatment available",
            "AudioFile": ""  # Placeholder for audio file, will be set later
        })

    return matched_entities

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    audio_file = request.files['file']
    
    # Save the uploaded file to a temporary location
    if audio_file:
        filename = secure_filename(audio_file.filename)
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file.write(audio_file.read())
            temp_audio_file_path = temp_audio_file.name

        # Transcribe the audio
        transcript = transcribe_audio(temp_audio_file_path)
        
        # Extract symptoms and other information
        entities = extract_entities(transcript)
        
        # Clean up the temporary file
        os.remove(temp_audio_file_path)
        
        return jsonify(entities)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
