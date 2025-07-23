
# 🧠 Hallucination Detection in Large Language Models (LLMs)

This project investigates hallucination detection in Large Language Models (LLMs) using both traditional machine learning and advanced LLM-based methods, enhanced with retrieval-augmented generation (RAG). The core objective is to detect when LLMs confidently generate factually incorrect or unverifiable information — a major challenge in trustworthy AI.

---

## 📌 Project Highlights

- **Dataset**: [FEVER](https://fever.ai/) — 145k+ human-generated claims with labeled evidence.
- **Models Compared**:
  - Traditional ML: Random Forest, XGBoost
  - LLMs: GPT-3.5, Claude (with and without RAG)
- **Key Techniques**:
  - Semantic similarity with Sentence Transformers
  - Named entity overlap with spaCy
  - RAG using FAISS vector store for evidence retrieval
  - Prompt engineering for LLM inference
- **Evaluation**: Class-wise accuracy, false positives, hallucination rate (especially for "Not Enough Info")

---

## 🧬 Dataset Overview

The FEVER dataset includes three classes:
- `SUPPORTS`: Claim is verified true based on evidence.
- `REFUTES`: Claim is directly contradicted by evidence.
- `NOT ENOUGH INFO`: Claim cannot be verified from available documents.

We performed:
- Class balancing (equal samples across labels)
- Data cleaning and formatting
- Stratified train/val/test split (70/15/15)

---

## 🔧 System Architecture

```text
                ┌────────────────────┐
                │   FEVER Dataset    │
                └────────┬───────────┘
                         │
         ┌───────────────▼───────────────┐
         │    Feature Engineering        │
         │  - Semantic Similarity        │
         │  - Entity Overlap             │
         │  - Text Length, etc.          │
         └───────────────┬───────────────┘
                         │
        ┌────────────────▼────────────────┐
        │       Parallel Model Pipelines  │
        │  ┌────────────┬──────────────┐  │
        │  │ ML Models  │ LLMs (w/ RAG)│  │
        │  └────────────┴──────────────┘  │
        └────────────────┬────────────────┘
                         │
                 ┌───────▼────────┐
                 │ Evaluation &   │
                 │ Hallucination  │
                 │ Detection      │
                 └────────────────┘
```

---

## 📈 Performance Summary

| Model                | Accuracy | With RAG | Hallucination Rate vs RF |
|---------------------|----------|----------|---------------------------|
| Random Forest        | 81.49%   | —        | —                         |
| XGBoost              | 81.88%   | —        | —                         |
| GPT-3.5 (standard)   | 59.40%   | ❌       | 87.04%                    |
| GPT-3.5 + RAG        | 66.00%   | ✅ +6.6% | —                         |
| Claude (standard)    | 36.60%   | ❌       | 95.06%                    |
| Claude + RAG         | 47.20%   | ✅ +10.6%| —                         |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/shubham-gaur-x/Hallucination-Detection-in-LLMs.git
cd Hallucination-Detection-in-LLMs
```

### 2. Setup environment
```bash
pip install -r requirements.txt
```

### 3. Run training pipelines
```bash
# For traditional ML
python train_ml_models.py

# For LLM with RAG
python run_rag_pipeline.py
```

---

## 🧠 Key Insights

- **Traditional ML > LLMs** in raw accuracy.
- **RAG significantly improves** LLM performance, especially Claude.
- **REFUTES class** remains hardest for LLMs, indicating limits in contradiction handling.
- Hybrid ML + LLM architectures hold potential for robust hallucination detection.

---

## 📚 Future Work

- Multi-hop RAG for deeper fact chaining
- Hybrid LLM + classifier ensemble systems
- Hallucination-specific loss functions
- Testing on other factual verification datasets (e.g., SciFact)

---

## 📄 License

This project is for academic/research purposes. Please cite appropriately if used.
