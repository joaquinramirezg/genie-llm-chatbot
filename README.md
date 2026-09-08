# GENIE: Spanish-language LLM chatbot for university regulations

> Undergraduate thesis project (B.S. Computer Science, UTEC). Fine-tuned Falcon-7B to answer student questions about university regulations in Spanish.
>
> **G.E.N.I.E.:** **G**enerative chatbot basado **e**n transformers especializado en la **n**ormativa y política de la Universidad de **I**ngeniería y T**e**cnología

**Co-author: Alexandra Shulca** · Live portfolio: [joaquinramirez.dev](https://joaquinramirez.dev/)

## The problem

University regulations are long, fragmented across many PDFs and web pages, and written in dense legal Spanish. The lack of a centralized way to get clear, updated information on policies, careers, syllabi, research projects, and services causes confusion in the UTEC community, impacts decision-making, and increases the workload of administrative staff. GENIE turns that UTEC-specific corpus into a conversational interface: ask a question in plain Spanish, get an answer grounded in the actual regulations.

## Data pipeline

1. **Scraping** (`scraping.py`): collected regulation documents from UTEC web domains and subdomains using BeautifulSoup.
2. **Normalization** (`convert-json.py`): converted raw documents into a clean JSON/CSV structure, using PyPDF2 to extract text from PDF files.
3. **Dataset construction** (`tesis_dataset.csv`): built an instruction-style Q&A dataset from the corpus with an LLM-assisted approach. Extracted text chunks were tokenized and sent to a language model API with a prompt instructing it to convert the text into single Q&A JSON pairs.

## Method

Two complementary notebooks:

- `tesis_falcon_7b_finetune.ipynb`: **LoRA fine-tuning of Falcon-7B-Instruct** on the Spanish Q&A dataset. Trained on Google Colab with an NVIDIA V100 Tensor Core GPU (16 GB GPU RAM, 51 GB total RAM). Hyperparameter optimization via Bayesian search across 20 trials, testing parameters like epochs (1 to 3) and 8-bit optimizers (`paged_adamw_8bit`).
- `tesis_embeddings_falcon.ipynb`: embedding-based retrieval over the regulation corpus, grounding answers in source text with a retrieval-then-generate approach. Combines the `E5-large-v2` embedding model with a Weaviate vector index and LangChain.

**Why fine-tune instead of prompt-only?** Before fine-tuning, base Falcon-7B lacked the institutional context to answer domain questions, generating incoherent or repetitive hallucinations (for example, outputting "¡Intercambio! ¡Intercambio! ¡Intercambio! ¡Intercambio!" when asked about the benefits of a student exchange).

## Results

| Metric | Base Falcon-7B | Fine-tuned |
|---|---|---|
| Eval loss | 2.916 | **0.46** |
| Perplexity | 18.471 | **1.58** |

Manual user tests ("pruebas de usuario manuales") were fundamental to measuring the model's factual accuracy. The fine-tuned model showed a substantial qualitative improvement, generating coherent, contextually accurate answers to UTEC-specific questions that previously confused the base model.

## Limitations

- Loss and perplexity measure fit to the dataset, not factual correctness of answers, which makes the subjective manual evaluation a necessary dependency.
- The model can hallucinate regulation details outside the training corpus; the embeddings-based grounding mitigates but does not eliminate this.
- The corpus is a snapshot; regulations change, and future iterations require an automated, continuous web scraping system to maintain accuracy.
- Production deployment requires rigorous data privacy and security analysis to protect sensitive educational information and ensure compliance with institutional regulations.

## Repo structure

```text
data/                          # processed dataset files
downloads/                     # raw scraped documents
reglamentos/                   # regulation source documents
scraping.py                    # scraper
convert-json.py                # normalization
tesis_falcon_7b_finetune.ipynb # LoRA fine-tuning
tesis_embeddings_falcon.ipynb  # embeddings + retrieval
tesis_dataset.csv              # training dataset
```

---

*This was my undergraduate thesis (2023 to 2024). My production GenAI work since then (agentic systems and RAG serving 1300+ organizations) lives in a private company GitLab; case studies are on [joaquinramirez.dev](https://joaquinramirez.dev/).*
