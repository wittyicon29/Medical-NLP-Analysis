# Medical Text Analysis Platform

## 📌 Overview
This application is a **Medical Text Analysis Platform** built with **Streamlit**, **LangChain**, and **Google Gemini**. It processes medical conversation transcripts, extracts key information, and generates structured reports using **Named Entity Recognition (NER)**, **Sentiment & Intent Analysis**, and **SOAP Notes generation**.

---

## 🚀 Features
- **Extracts Medical Entities** (Symptoms, Diagnosis, Treatment, Prognosis) using **NER-BERT, Regex**.
- **Performs Sentiment & Intent Analysis** using a few shot tuned **Clinical BERT model**.
- **Generates SOAP Notes** (Subjective, Objective, Assessment, Plan) using **Google Gemini**.
- **Processes text from direct input or PDF files**.

---

## 🛠️ Setup & Installation

### 1️⃣ Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Virtual Environment (Recommended)
- Streamlit
- Required Python Libraries

### 2️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd medical-text-analysis
```

### 3️⃣ Install Dependencies
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add your **Google Gemini API Key**:
```env
GOOGLE_API_KEY=<your-api-key>
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

The application will launch in your default browser.

---

## 🖼️ Sample Output Screenshots
(Add images after running the app)

1. **Home Screen:** (Upload PDF / Enter Text)
2. **NER Analysis Results:** (Extracted Entities)
3. **Sentiment & Intent Analysis:** (BERT-based Results)
4. **SOAP Notes Output:** (Structured Report)

---

## 🔍 Methodologies Used

### 1️⃣ Named Entity Recognition (NER)
- Uses **MedSpacy** for clinical entity extraction.
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic similarity matching.
- **BART Large-CNN** for summarization.

### 2️⃣ Sentiment & Intent Analysis
- **BERT-based model** for text classification.
- Trained on medical dialogues for patient **sentiment** (Anxious, Neutral, Reassured) and **intent** (Seeking Reassurance, Reporting Symptoms, Expressing Concern).

### 3️⃣ SOAP Notes Generation
- **LangChain** with **Google Gemini (gemini-1.5-pro)** for AI-driven SOAP structuring.
- Uses **Prompt Engineering** to format structured medical notes.

---

## 📜 File Structure
```
📂 medical-text-analysis/
├── app.py                # Streamlit Application
├── NER.py                # Named Entity Recognition Logic
├── Sentiment_Intent.py   # Sentiment & Intent Analysis
├── SOAP.py               # SOAP Note Generation
├── requirements.txt      # Dependencies
└── README.md             # Documentation (You are here)
```

---

## 🏥 Use Cases
- **Medical Research:** Extract structured data from patient conversations.
- **Clinical Documentation:** Automate SOAP note generation.
- **AI-assisted Diagnosis:** Support decision-making in healthcare.

---

## 📄 License
MIT License - Free to use and modify.
