# 🤖 IntelliAnalyze: A Dual-Mode RAG & Pandas Data Agent

> **Bringing the power of agentic workflows to your data, whether it's in a spreadsheet or a document.**

---

## 📝 Detailed Overview

**IntelliAnalyze** is an advanced AI Data Analyst built to bridge the gap between structured calculations and unstructured knowledge retrieval. Using a sophisticated **LangGraph Orchestrator**, the agent intelligently routes user queries between two specialized execution paths:

1.  **📊 Structured Data Mode (Pandas)**: For numerical analysis, trends, aggregations, and generating dynamic visualizations from CSV/Excel files. It features a self-healing loop that writes, executes, and repairs Python code locally.
2.  **📄 Unstructured Search Mode (FAISS RAG)**: For deep context retrieval from PDF knowledge bases. It uses FAISS vector stores and semantic search to answer complex questions based on your documents.

The core "brain" uses a routing logic to handle **Hybrid Queries**, allowing you to compare live data trends against document-based policies or notes in a single conversation.

---

## 📺 Project Walkthrough & Screenshots

For a complete step-by-step visual guide, UI demonstrations, and detailed logs of the agent's logic in action, please visit my official documentation:

👉 **[View Detailed Project Walkthrough (Google Docs)](https://docs.google.com/document/d/1RG8U72vuvb8ynw2BGX-ncoO_VHhy1WL9QHwFGznYYvg/edit?usp=sharing)**

*This document contains visual proof of the UI, error-handling logs, and end-to-end user flows.*

---

## 🛠️ Core Technical Stack

| Component | Technology |
| :--- | :--- |
| **LLM Engine** | ✨ Google Gemini-2.0-Flash (with 1.5-Flash fallback) |
| **Orchestration** | 🧠 LangGraph & LangChain |
| **Vector Search** | 🔍 FAISS (Facebook AI Similarity Search) |
| **Data Handling** | 📈 Pandas, NumPy |
| **Interface** | 🎨 Streamlit (Premium Dark Theme) |
| **Environment** | ⚙️ Python (PIP, Dotenv, VS Code Debugging) |

### 🧠 Logic & Engineering Skills
This project demonstrates advanced software engineering principles including:
-   **Regex (Regular Expressions)**: Used for precise data cleaning and pattern matching in unstructured text.
-   **JSON Management**: Efficient data exchange between the LLM and the local code runner.
-   **Validation Logic**: Implementation of complex rules like **Leap Year detection** and **Modulus math** for robust date/data validation.

---

## 📚 Deepali's Learning Update
*This project marks the culmination of an intensive Python & AI learning journey.*

My progress evolved rapidly from foundational concepts to advanced agentic architecture:
-   **The Basics**: Mastered Python fundamentals including Data Types (**List, Tuple, Dictionary**) and flow control.
-   **Advanced OOP**: Implemented **Polymorphism** and **Encapsulation** to build reusable agent components.
-   **Algorithmic Thinking**: Utilized **Recursion** for complex problem solving and **Memoryview** for high-efficiency data handling.
-   **Pro Tooling**: Hands-on expertise in **VS Code debugging**, **PIP package management**, and environment orchestration.

---

## 🚀 Installation & Setup

### 1. Prerequisite
Obtain a **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone and Install
```bash
# Clone the repository
git clone https://github.com/deep-dev-12489/IntelliAnalyze.git
cd IntelliAnalyze

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 🛡️ License
This project is licensed under the MIT License - see the LICENSE file for details.
