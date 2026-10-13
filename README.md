# AI-Powered Multilingual Legal Judgment Explanation System

An AI-powered application that helps users understand complex legal judgments through simplified explanations, legal provision analysis, multilingual support, and case-continuation insights.

## Overview

Legal judgments often contain complex terminology, lengthy explanations, and references to multiple legal provisions. Understanding these documents can be difficult for people without a legal background.

The **AI-Powered Multilingual Legal Judgment Explanation System** aims to make legal information easier to understand by analyzing uploaded judgments, generating summaries, identifying relevant legal provisions, and explaining legal content in the user's preferred language.

The system also provides preliminary insights into issues that may be relevant to subsequent hearings in a continuing case.

## Key Features

- **PDF Judgment Upload:** Upload a legal judgment in PDF format for analysis.
- **Judgment Summarization:** Generate concise summaries of lengthy legal documents.
- **Legal Provision Extraction:** Identify legal provisions and section references mentioned in a judgment.
- **Provision Validation and Ranking:** Validate extracted legal references and rank potentially relevant provisions.
- **Multilingual Explanation:** Explain legal content in the user's preferred language, including Tamil where supported.
- **Case Continuation Insights:** Identify issues that may require further consideration in subsequent hearings.
- **Related Legal Provisions:** Present potentially related provisions to support further legal research.
- **Interactive Web Interface:** Access the system through a frontend connected to the backend API.

## System Architecture

```text
             User
              |
              v
       React Web Interface
              |
              v
         FastAPI Backend
              |
              v
       PDF Text Extraction
              |
              v
      Judgment Preprocessing
              |
              v
       Legal Provision Analysis
              |
              v
     Semantic Search / Retrieval
              |
              v
       AI-Powered Processing
              |
       +------+------+
       |             |
       v             v
  Simplified      Multilingual
  Summary         Explanation
       |
       v
 Case Continuation Insights
       |
       v
  Results in Web Interface
```

## Technology Stack

### Frontend
- React
- Vite
- JavaScript
- HTML and CSS

### Backend
- Python
- FastAPI
- Uvicorn

### Artificial Intelligence and NLP
- Google Gemini API for supported language-generation tasks
- Sentence Transformers for text embeddings
- `all-MiniLM-L6-v2` embedding model
- FAISS for similarity search and retrieval

### Document Processing
- PDF text extraction
- Text preprocessing and chunking
- Legal provision extraction and normalization
- Semantic retrieval

## Project Structure

```text
AI-Legal-Judgment-System/
├── backend/
│   └── app/
│       ├── api/
│       ├── services/
│       └── main.py
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── index.html
├── data/
│   └── evaluation/
├── scripts/
├── tests/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

*Note: The displayed structure is representative. Retain the folders and filenames that actually exist in your repository.*

## Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible supported version
- Node.js and npm
- Git
- A valid Google Gemini API key if using the Gemini provider

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Legal-Judgment-System.git
cd AI-Legal-Judgment-System
```

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository URL.

### 2. Create and activate a Python virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate local execution-policy setting or activate the environment through Command Prompt.

### 3. Install backend dependencies

```powershell
pip install -r requirements.txt
```

If your project uses additional dependency files, install the dependencies documented in those files as well.

### 4. Configure environment variables

Create a local `.env` file in the project root using the configuration required by your application.

Example:

```env
GEMINI_API_KEY=your_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=your_supported_gemini_model
LOCAL_LLM_MODEL=qwen2.5:1.5b
```

Replace the example values with your actual configuration. Use a supported model name and configure a local model only if your implementation requires it.

**Security:** Never commit your `.env` file, API keys, passwords, or other credentials to GitHub. Keep `.env.example` free of real secrets.

### 5. Start the backend

Run the command from the project root:

```powershell
uvicorn backend.app.main:app --reload
```

The backend should be available at:

- API base URL: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### 6. Install frontend dependencies

Open a second terminal and navigate to the frontend:

```powershell
cd frontend
npm install
```

### 7. Start the frontend

```powershell
npm run dev
```

Open the local URL printed by Vite, typically:

http://localhost:5173

Keep both the backend and frontend terminals running while using the application.

## How to Use

1. Start the backend and frontend.
2. Open the application in your browser.
3. Upload a supported legal judgment PDF.
4. Run the judgment analysis.
5. Review the generated summary and extracted legal provisions.
6. Examine the ranked or related provisions returned by the system.
7. Use the multilingual explanation feature where available.
8. Review case-continuation insights as preliminary information for further legal research.

The available features and output fields depend on the current implementation and the selected judgment.

## Data and Evaluation

The project includes evaluation resources intended to support testing of legal judgment analysis and retrieval. These may include test cases, expected summaries, expected legal sections, and legal question-answering data.

Evaluation should assess relevant aspects such as:

- Summary relevance and factual consistency
- Accuracy of extracted legal provisions
- Retrieval relevance
- Quality of multilingual explanations
- Correctness and usefulness of generated insights

Results should be measured using a documented test set. Do not claim a particular accuracy percentage unless it has been measured and verified.

## Responsible Use and Limitations

This project is intended for **legal information and educational assistance**, not as a replacement for a qualified lawyer.

- AI-generated summaries and explanations may contain errors or omit important context.
- Extracted legal provisions and suggested provisions must be verified against authoritative legal sources.
- Case-continuation insights are preliminary research assistance, not guaranteed legal strategies or predictions.
- The system does not replace professional legal advice or establish the correct defence for a particular case.
- Avoid uploading confidential, sensitive, or personally identifiable legal documents to services unless you are authorized to do so and understand the applicable privacy implications.
- Generated explanations should preserve the original judgment's meaning and context.

## Future Enhancements

- Support for additional Indian languages.
- Improved citation and legal provision verification.
- Better retrieval and source-grounded explanations.
- Evaluation using legal-domain benchmarks and expert review.
- Improved document handling and privacy controls.
- More detailed references linking generated explanations to relevant judgment passages.

## Contribution

Contributions, suggestions, and bug reports are welcome. Before submitting changes, test the relevant functionality and ensure that API keys, confidential documents, virtual environments, generated model files, and other sensitive data are excluded from version control.

## License

Choose and add an appropriate open-source license before distributing the project. Ensure that any datasets, judgments, dependencies, and other included materials may legally be redistributed under their respective terms.

---

**Disclaimer:** This application provides AI-generated legal information for educational and research purposes only. It does not constitute legal advice. Consult a qualified legal professional for advice concerning a specific case.
