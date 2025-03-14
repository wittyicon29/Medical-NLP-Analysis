import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

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

# Example usage
if __name__ == "__main__":
    transcript = """
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
    
    soap_note = generate_soap_note(transcript)
    print(json.dumps(soap_note, indent=2))