import medspacy
from medspacy.ner import TargetRule
import re
import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def analyze_medical_text(text: str) -> dict:
    """
    Analyzes medical text to extract structured information.
    
    Args:
        text (str): Input medical text/conversation
        
    Returns:
        dict: Structured medical information in JSON format
    """
    nlp = medspacy.load()
    similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Extract patient name
    def extract_patient_name(text):
        match = re.search(r"(?:Mr\.|Ms\.|Patient)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        return match.group(1) if match else "Unknown"

    # Setup target rules
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [
        TargetRule("neck pain", "PROBLEM"),
        TargetRule("back pain", "PROBLEM"),
        TargetRule("headache", "PROBLEM"),
        TargetRule("whiplash injury", "DIAGNOSIS"),
        TargetRule("physiotherapy", "PROCEDURE"),
        TargetRule("painkillers", "MEDICATION"),
        TargetRule("full recovery", "PROGNOSIS")
    ]
    target_matcher.add(target_rules)

    # Initialize structured data
    structured_data = {
        "Patient_Name": extract_patient_name(text),
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": [],
        "Current_Status": [],
        "Prognosis": []
    }

    # Process text
    doc = nlp(text)
    entity_mapping = {
        "PROBLEM": "Symptoms",
        "DIAGNOSIS": "Diagnosis",
        "MEDICATION": "Treatment",
        "PROCEDURE": "Treatment",
        "PROGNOSIS": "Prognosis"
    }

    # Extract entities
    extracted_entities = set()
    for ent in doc.ents:
        category = entity_mapping.get(ent.label_, None)
        if category:
            structured_data[category].append(ent.text)
            extracted_entities.add(ent.text.lower())

    # Standard medical terms for semantic matching
    standard_medical_terms = {
        "Symptoms": ["headache", "dizziness", "nausea", "blurred vision", "fatigue"],
        "Diagnosis": ["concussion", "migraine", "brain injury", "whiplash"],
        "Treatment": ["ibuprofen", "paracetamol", "rest", "physical therapy"],
        "Prognosis": ["full recovery", "gradual improvement"]
    }

    # Semantic similarity matching
    tokens = text.split()
    for token in tokens:
        if token.lower() in extracted_entities:
            continue
        
        best_match = None
        best_score = -1
        best_category = None

        for category, terms in standard_medical_terms.items():
            for term in terms:
                score = util.cos_sim(
                    similarity_model.encode(token.lower()),
                    similarity_model.encode(term)
                ).item()
                if score > best_score and score > 0.65:
                    best_score = score
                    best_match = term
                    best_category = category

        if best_match and best_category:
            structured_data[best_category].append(best_match)

    # Generate summary
    summary_output = summarizer(text, max_length=200, min_length=50, do_sample=False)
    final_summary = summary_output[0]['summary_text']

    # Prepare final output
    structured_summary = {
        "Patient_Name": structured_data["Patient_Name"],
        "Symptoms": list(set(structured_data["Symptoms"])),
        "Diagnosis": list(set(structured_data["Diagnosis"])),
        "Treatment": list(set(structured_data["Treatment"])),
        "Current_Status": ["Ongoing symptoms present"] if structured_data["Symptoms"] else ["No active symptoms"],
        "Prognosis": list(set(structured_data["Prognosis"])),
        "Summary": final_summary
    }

    return structured_summary

# Example usage
if __name__ == "__main__":
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today? 
    Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then. 
    Physician: I understand you were in a car accident last September. Can you walk me through what 
    happened? 
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle 
    Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, 
    which pushed my car into the one in front. 
    Physician: That sounds like a strong impact. Were you wearing your seatbelt? 
    Patient: Yes, I always do. 
    Physician: What did you feel immediately after the accident? 
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I 
    could feel pain in my neck and back almost right away. 
    Physician: Did you seek medical attention at that time? 
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a 
    whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home. 
    Physician: How did things progress after that? 
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble 
    sleeping and had to take painkillers regularly. It started improving after that, but I had to go through 
    ten sessions of physiotherapy to help with the stiffness and discomfort. 
    Physician: That makes sense. Are you still experiencing pain now? 
    Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though. 
    Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or 
    difficulty concentrating? 
    Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues 
    from the accident. 
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that? 
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really 
    stopped me from doing anything. 
    Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility 
    and any lingering pain. 
    [Physical Examination Conducted] 
    Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no 
    tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition. 
    Patient: That’s a relief! 
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to 
    make a full recovery within six months of the accident. There are no signs of long-term damage or 
    degeneration. 
    Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future? 
    Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything 
    changes or you experience worsening symptoms, you can always come back for a follow-up. But at 
    this point, you’re on track for a full recovery. 
    Patient: Thank you, doctor. I appreciate it. 
    Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need 
    anything.
    """
    result = analyze_medical_text(sample_text)
    print(json.dumps(result, indent=4))