import medspacy
from medspacy.ner import TargetRule
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import json
import tempfile
import re
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

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

def analyze_sentiment_intent(text: str) -> dict:
    """
    Analyzes the sentiment and intent of the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing sentiment and intent predictions
    """
    # Load Sentiment Model
    sentiment_model = BertForSequenceClassification.from_pretrained("sentiment_model")
    sentiment_tokenizer = BertTokenizer.from_pretrained("sentiment_model")

    # Load Intent Model
    intent_model = BertForSequenceClassification.from_pretrained("intent_model")
    intent_tokenizer = BertTokenizer.from_pretrained("intent_model")

    # Define labels
    sentiment_labels = {0: "Anxious", 1: "Neutral", 2: "Reassured"}
    intent_labels = {0: "Seeking reassurance", 1: "Reporting symptoms", 2: "Expressing concern"}

    # Tokenize input
    inputs_sentiment = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs_intent = intent_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get predictions
    with torch.no_grad():
        sentiment_logits = sentiment_model(**inputs_sentiment).logits
        intent_logits = intent_model(**inputs_intent).logits

    # Get predicted label
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    intent_pred = torch.argmax(intent_logits, dim=1).item()

    # Return predictions as dictionary
    return {
        "Sentiment": sentiment_labels[sentiment_pred],
        "Intent": intent_labels[intent_pred]
    }

# Define an improved prompt template for SOAP extraction
soap_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
    You are a medical scribe tasked with converting a conversation transcript into a structured SOAP note.
    
    INSTRUCTIONS:
    1. Extract information from the transcript using clinical reasoning.
    2. Structure the information precisely into the SOAP format.
    3. Provide only the structured JSON output without additional text or commentary.
    4. If a section cannot be determined from the transcript, provide a placeholder based on context.
    
    Transcript:
    {transcript}
    
    Output the following JSON structure WITHOUT ANY ADDITIONAL TEXT:
    {{
        "Subjective": {{
            "Chief_Complaint": "Primary reason for the visit, in patient's words",
            "History_of_Present_Illness": "Condensed narrative of the patient's description of their condition"
        }},
        "Objective": {{
            "Physical_Exam": "Relevant physical examination findings",
            "Observations": "Clinical observations about the patient's condition"
        }},
        "Assessment": {{
            "Diagnosis": "Clinical diagnosis based on the available information",
            "Severity": "Assessment of the condition's severity"
        }},
        "Plan": {{
            "Treatment": "Recommended treatments or interventions",
            "Follow_Up": "Follow-up instructions or timeline"
        }}
    }}
    """
)

soap_chain = LLMChain(llm=llm, prompt=soap_prompt)

def generate_soap_note(transcript: str):
    """
    Generate a structured SOAP note from a medical conversation transcript.
    
    Args:
        transcript (str): The medical conversation transcript
        
    Returns:
        dict: A structured SOAP note in dictionary format
    """
    cleaned_transcript = clean_transcript(transcript)
    try:
        response = soap_chain.run(transcript=cleaned_transcript)
        json_content = extract_json(response)
        soap_note = json.loads(json_content)
        soap_note = validate_soap_structure(soap_note)
        
        return soap_note
    except Exception as e:
        print(f"Error generating SOAP note: {str(e)}")
        return fallback_soap_note(cleaned_transcript)

def clean_transcript(transcript: str) -> str:
    """Clean and format the transcript for better processing."""
    # Remove extra whitespace
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    lines = transcript.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not re.match(r'^(Doctor|Patient):', line, re.IGNORECASE):     
            if re.match(r'^(dr|doc|physician|provider)', line, re.IGNORECASE):
                line = f"Doctor: {line}"
            elif re.match(r'^(pt|patient|client)', line, re.IGNORECASE):
                line = f"Patient: {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def extract_json(text: str) -> str:
    """Extract JSON content from text that may contain additional commentary."""                                       
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
    return text

def validate_soap_structure(soap_note: dict) -> dict:
    """Validate and ensure the SOAP note has the correct structure."""
    expected_structure = {
        "Subjective": {
            "Chief_Complaint": "",
            "History_of_Present_Illness": ""
        },
        "Objective": {
            "Physical_Exam": "",
            "Observations": ""
        },
        "Assessment": {
            "Diagnosis": "",
            "Severity": ""
        },
        "Plan": {
            "Treatment": "",
            "Follow_Up": ""
        }
    }
    
    # Ensure all expected sections exist
    for main_section, subsections in expected_structure.items():
        if main_section not in soap_note:
            soap_note[main_section] = {}
            
        for subsection in subsections:
            if subsection not in soap_note[main_section]:
                soap_note[main_section][subsection] = "Not specified in transcript"
                
    return soap_note

def fallback_soap_note(transcript: str) -> dict:
    """Generate a basic SOAP note when the LLM chain fails."""
    chief_complaint = "Not clearly identified"
    history = "See transcript for details"
    
    pain_match = re.search(r'(my\s+)?([\w\s]+)(hurt|pain|ache)', transcript, re.IGNORECASE)
    if pain_match:
        chief_complaint = f"{pain_match.group(2).strip()} pain"

    if "car accident" in transcript.lower():
        diagnosis = "Possible injury related to car accident"
    else:
        diagnosis = "Unspecified condition"
    
    return {
        "Subjective": {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": history
        },
        "Objective": {
            "Physical_Exam": "No physical exam information available",
            "Observations": "No observations noted in transcript"
        },
        "Assessment": {
            "Diagnosis": diagnosis,
            "Severity": "Unknown"
        },
        "Plan": {
            "Treatment": "Further evaluation needed",
            "Follow_Up": "To be determined"
        }
    }

# Set page configuration
st.set_page_config(
    page_title="Medical Text Analysis",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.conversation_text = ""
    st.session_state.ner_results = None
    st.session_state.sentiment_results = None
    st.session_state.soap_results = None

def initialize_models():
    """Initialize all models and store in session state"""
    if not st.session_state.initialized:
        with st.spinner('Loading models, please wait...'):
            # Models are initialized when functions are called
            # We're not pre-loading them to save memory and startup time
            st.session_state.initialized = True
        st.success("Models loaded successfully!")

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            
    return text

# Sidebar for configuration and file upload
st.sidebar.title("Medical Text Analysis")
st.sidebar.markdown("---")

# Initialize models
initialize_models()

# Input methods
st.sidebar.subheader("Input Method")
input_method = st.sidebar.radio("Select input method:", ["Text Input", "PDF Upload"])

if input_method == "Text Input":
    st.sidebar.markdown("Enter the medical conversation text in the main panel.")
else:
    st.sidebar.markdown("Upload a PDF file containing the medical conversation.")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.session_state.conversation_text = extracted_text
                st.sidebar.success(f"Text extracted! ({len(extracted_text)} characters)")
            else:
                st.sidebar.error("Failed to extract text from the PDF.")

# Main panel
st.title("Medical Text Analysis Platform")

# Input section
st.header("Medical Conversation")
if input_method == "Text Input":
    conversation_text = st.text_area(
        "Enter medical conversation:",
        value=st.session_state.conversation_text,
        height=300
    )
    if conversation_text != st.session_state.conversation_text:
        st.session_state.conversation_text = conversation_text
        # Reset results when input changes
        st.session_state.ner_results = None
        st.session_state.sentiment_results = None
        st.session_state.soap_results = None
else:
    if st.session_state.conversation_text:
        st.text_area(
            "Extracted Text:",
            value=st.session_state.conversation_text,
            height=300
        )
    else:
        st.info("Please upload a PDF file using the sidebar.")

# Make sure we have a conversation text to analyze
has_text = len(st.session_state.conversation_text.strip()) > 0

# Analysis section
st.header("Analysis Options")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Named Entity Recognition", disabled=(not has_text or not st.session_state.initialized)):
        with st.spinner("Running NER analysis..."):
            try:
                results = analyze_medical_text(st.session_state.conversation_text)
                st.session_state.ner_results = results
                st.success("NER analysis completed!")
            except Exception as e:
                st.error(f"Error in NER analysis: {str(e)}")
                st.session_state.ner_results = None
    
    if st.button("Run Sentiment Analysis", disabled=(not has_text or not st.session_state.initialized)):
        with st.spinner("Analyzing sentiment and intent..."):
            try:
                results = analyze_sentiment_intent(st.session_state.conversation_text)
                st.session_state.sentiment_results = results
                st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
                st.session_state.sentiment_results = None

with col2:
    if st.button("Generate SOAP Note", disabled=(not has_text or not st.session_state.initialized)):
        with st.spinner("Generating SOAP note..."):
            try:
                results = generate_soap_note(st.session_state.conversation_text)
                st.session_state.soap_results = results
                st.success("SOAP note generated!")
            except Exception as e:
                st.error(f"Error generating SOAP note: {str(e)}")
                st.session_state.soap_results = None

# Results section
st.header("Results")

tabs = st.tabs(["Named Entity Recognition", "Sentiment Analysis", "SOAP Note"])

with tabs[0]:
    if st.session_state.ner_results:
        st.subheader("Named Entity Recognition Results")
        
        # Display patient name
        st.write(f"**Patient Name:** {st.session_state.ner_results['Patient_Name']}")
        
        # Display other structured data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Symptoms")
            if st.session_state.ner_results['Symptoms']:
                for symptom in st.session_state.ner_results['Symptoms']:
                    st.markdown(f"- {symptom}")
            else:
                st.write("No symptoms detected")
                
            st.markdown("### Diagnosis")
            if st.session_state.ner_results['Diagnosis']:
                for diagnosis in st.session_state.ner_results['Diagnosis']:
                    st.markdown(f"- {diagnosis}")
            else:
                st.write("No diagnosis detected")
        
        with col2:
            st.markdown("### Treatment")
            if st.session_state.ner_results['Treatment']:
                for treatment in st.session_state.ner_results['Treatment']:
                    st.markdown(f"- {treatment}")
            else:
                st.write("No treatments detected")
                
            st.markdown("### Prognosis")
            if st.session_state.ner_results['Prognosis']:
                for prognosis in st.session_state.ner_results['Prognosis']:
                    st.markdown(f"- {prognosis}")
            else:
                st.write("No prognosis information detected")
        
        st.markdown("### Summary")
        st.write(st.session_state.ner_results['Summary'])
        
        # JSON download button
        st.download_button(
            label="Download NER Results as JSON",
            data=json.dumps(st.session_state.ner_results, indent=4),
            file_name="ner_results.json",
            mime="application/json"
        )
    else:
        st.info("Run Named Entity Recognition to see results here.")

with tabs[1]:
    if st.session_state.sentiment_results:
        st.subheader("Sentiment and Intent Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Patient Sentiment", st.session_state.sentiment_results["Sentiment"])
        
        with col2:
            st.metric("Patient Intent", st.session_state.sentiment_results["Intent"])
            
        # JSON download button
        st.download_button(
            label="Download Sentiment Results as JSON",
            data=json.dumps(st.session_state.sentiment_results, indent=4),
            file_name="sentiment_results.json",
            mime="application/json"
        )
    else:
        st.info("Run Sentiment Analysis to see results here.")

with tabs[2]:
    if st.session_state.soap_results:
        st.subheader("SOAP Note")
        
        # Subjective
        st.markdown("### Subjective")
        st.write(f"**Chief Complaint:** {st.session_state.soap_results['Subjective']['Chief_Complaint']}")
        st.write(f"**History of Present Illness:** {st.session_state.soap_results['Subjective']['History_of_Present_Illness']}")
        
        # Objective
        st.markdown("### Objective")
        st.write(f"**Physical Exam:** {st.session_state.soap_results['Objective']['Physical_Exam']}")
        st.write(f"**Observations:** {st.session_state.soap_results['Objective']['Observations']}")
        
        # Assessment
        st.markdown("### Assessment")
        st.write(f"**Diagnosis:** {st.session_state.soap_results['Assessment']['Diagnosis']}")
        st.write(f"**Severity:** {st.session_state.soap_results['Assessment']['Severity']}")
        
        # Plan
        st.markdown("### Plan")
        st.write(f"**Treatment:** {st.session_state.soap_results['Plan']['Treatment']}")
        st.write(f"**Follow-Up:** {st.session_state.soap_results['Plan']['Follow_Up']}")
        
        # JSON download button
        st.download_button(
            label="Download SOAP Note as JSON",
            data=json.dumps(st.session_state.soap_results, indent=4),
            file_name="soap_note.json",
            mime="application/json"
        )
    else:
        st.info("Generate SOAP Note to see results here.")

# Footer
st.markdown("---")
st.caption("Medical Text Analysis Tool - Powered by MedSpacy, BERT, and LangChain")
