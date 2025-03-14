# Medical Text Analysis Platform

## 📌 Overview
This application is a **Medical Text Analysis Platform** built with **Streamlit**, **LangChain**, and **Google Gemini**. It processes medical conversation transcripts, extracts key information, and generates structured reports using **Named Entity Recognition (NER)**, **Sentiment & Intent Analysis**, and **SOAP Notes generation**.

---

## 🚀 Features
- **Extracts Medical Entities** (Symptoms, Diagnosis, Treatment, Prognosis) using **NER-BERT, Regex**.
- **Performs Sentiment & Intent Analysis** using a few-shot tuned **Clinical BERT model**.
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

## 🗀️ Sample Output Screenshots

1. **Home Screen:** (Upload PDF / Enter Text)
   ![Screenshot 2025-03-14 165917](https://github.com/user-attachments/assets/f9c8f55b-8be0-4a82-866e-96ad44505682)

2. **NER Analysis Results:** (Extracted Entities)
   ![Screenshot 2025-03-14 171941](https://github.com/user-attachments/assets/dda61754-cb81-4b3f-9aff-244e3d1ffbe8)

3. **Sentiment & Intent Analysis:** (BERT-based Results)
   ![Screenshot 2025-03-14 171121](https://github.com/user-attachments/assets/26d7c10b-7d00-4cdc-89e9-604407ed4df6)

4. **SOAP Notes Output:** (Structured Report)
   ![Screenshot 2025-03-14 171021](https://github.com/user-attachments/assets/03d6f501-be41-48d8-a998-04430aa9166b)


---

## 🔍 Methodologies Used

### 1️⃣ Named Entity Recognition (NER)
- Uses **MedSpacy** for clinical entity extraction.
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic similarity matching.
- **BART Large-CNN** for summarization.
- **The NER can take a long time depending on the transcript length as it is utilizing CPU**.

### 2️⃣ Sentiment & Intent Analysis
- **BERT-based model** for text classification.
- Trained on medical dialogues for patient **sentiment** (Anxious, Neutral, Reassured) and **intent** (Seeking Reassurance, Reporting Symptoms, Expressing Concern).

### 3️⃣ SOAP Notes Generation
- **LangChain** with **Google Gemini (gemini-1.5-pro)** for AI-driven SOAP structuring.
- Uses **Prompt Engineering** to format structured medical notes.

---

## 📝 File Structure
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

## 📄 Additional Insights

### **How would you handle ambiguous or missing medical data in the transcript?**
- Use **rule-based heuristics** to detect missing data (e.g., missing symptoms, partial medications).
- Leverage **zero-shot/few-shot learning** using **LLMs** to infer missing details based on context.
- Prompt users to clarify missing information using **interactive querying**.

### **What pre-trained NLP models would you use for medical summarization?**
- **BART Large-CNN** (for abstractive summarization).
- **BioBERT / ClinicalBERT** (domain-specific summarization in healthcare).
- **Pegasus (Google AI)** (state-of-the-art summarization model fine-tuned for medical literature).
- **T5 (Text-to-Text Transfer Transformer)** (for structured and query-based summarization).

### **How would you fine-tune BERT for medical sentiment detection?**
- Collect **annotated medical conversations** with labeled sentiment.
- Use a **pre-trained ClinicalBERT model** and fine-tune it using transfer learning.
- Apply **data augmentation techniques** (e.g., back translation, paraphrasing) to enrich training data.
- Train on **balanced datasets** to avoid class imbalances affecting accuracy.
- Use **weighted loss functions** (e.g., Focal Loss) to handle skewed class distributions.

### **What datasets would you use for training a healthcare-specific sentiment model?**
- **I2B2 Clinical Notes Dataset** (annotated clinical dialogues).
- **MIMIC-III / MIMIC-IV** (Medical ICU records with textual notes).
- **MedDialog** (large dataset of doctor-patient conversations).
- **PubMed abstracts** (for medical intent classification).
- **Emory Sentiment Dataset** (for fine-tuning sentiment-based classification).

### **How would you train an NLP model to map medical transcripts into SOAP format?**
- **Supervised Learning:** Train an NLP model on manually labeled SOAP notes using MIMIC-III.
- **Few-shot Prompt Engineering:** Use LLMs (like Google Gemini) with structured prompts to auto-generate SOAP sections.
- **Fine-tune a Transformer-based Model:** Use **T5** or **GPT-3.5/4** with a SOAP-formatted dataset.
- **Hybrid Approach:** Combine **rule-based systems (regex, heuristics)** with **deep learning** to ensure accuracy.

### **What rule-based or deep-learning techniques would improve the accuracy of SOAP note generation?**
- **Rule-Based:**
  - Use **regular expressions** to extract key medical phrases.
  - Apply **MedSpacy** for domain-specific text preprocessing.
- **Deep Learning-Based:**
  - Use **Sequence-to-Sequence models (T5, BART)** for structured generation.
  - Train **BERT-based extractive models** to identify SOAP components.
  - Fine-tune **GPT-4 / Gemini** for summarization and structured note formatting.

---

## 📝 License
MIT License - Free to use and modify.

