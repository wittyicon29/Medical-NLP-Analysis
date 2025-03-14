import streamlit as st
import json
import tempfile
import os
import re
from pathlib import Path
import PyPDF2

# Import your modules
from NER import analyze_medical_text
from Sentiment_Intent import analyze_sentiment_intent
from SOAP import generate_soap_note

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