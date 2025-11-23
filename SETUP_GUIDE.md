# RAG System Setup & Usage Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/)
- Windows PowerShell (or equivalent terminal)

### 2. Setup Steps

#### A. Environment is Already Configured! ✓
The virtual environment has been created and all dependencies installed.

#### B. Add Your API Key
1. Open the `.env` file in the project root
2. Replace `your-api-key-here` with your actual OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. Save the file

#### C. Verify Document Extraction ✓
The Wikipedia article on Artificial Intelligence has already been extracted to `Selected_Document.txt`
- 87,994 characters
- 181 paragraphs
- Covers AI history, techniques, applications, and ethics

### 3. Running the RAG System

```powershell
# Activate virtual environment (if not already active)
.venv\Scripts\activate

# Run the RAG application
python RAG_app.py
```

The system will:
1. Load the document from `Selected_Document.txt`
2. Split into chunks (takes a few seconds)
3. Generate embeddings (takes 10-30 seconds depending on hardware)
4. Build FAISS index
5. Load cross-encoder for re-ranking
6. Start interactive Q&A session

### 4. Using the System

Once running, you'll see:
```
======================================================================
RAG System Ready!
======================================================================
Enter 'exit' or 'quit' to end.

Your question:
```

Type your question and press Enter. Examples:
- "What is artificial intelligence and when was it founded?"
- "How do neural networks work in AI?"
- "What are the ethical concerns related to AI?"
- "What is deep learning?"
- "Who are the key figures in AI history?"

To exit, type `exit` or `quit`.

## Running Experiments

### Experiment 1: Default Configuration (Baseline)
```powershell
python RAG_app.py
```
Parameters: chunk_size=500, chunk_overlap=50

### Experiment 2: Large Chunks
1. Edit `RAG_app.py`, lines 34-35:
   ```python
   chunk_size = 1000
   chunk_overlap = 100
   ```
2. Save and run: `python RAG_app.py`
3. Ask the same questions and compare results

### Experiment 3: Small Chunks
1. Edit `RAG_app.py`, lines 34-35:
   ```python
   chunk_size = 250
   chunk_overlap = 25
   ```
2. Save and run: `python RAG_app.py`
3. Ask the same questions and compare results

### What to Document
For each configuration, record:
- Answer quality (completeness, accuracy, relevance)
- Response time (how long to generate answers)
- Any notable differences in the retrieved context
- Which configuration worked best for which types of questions

## Troubleshooting

### Error: "Import openai could not be resolved"
**Solution:** Activate the virtual environment:
```powershell
.venv\Scripts\activate
```

### Error: "OPENAI_API_KEY not found"
**Solution:** Make sure you've created `.env` file with your API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Error: "Selected_Document.txt not found"
**Solution:** Run the text extractor first:
```powershell
python text_extractor.py
```

### Error: 403 when fetching Wikipedia
**Solution:** The scraper includes proper headers. If still blocked, try:
1. Different Wikipedia article URL
2. Use a local PDF file instead (modify text_extractor.py)
3. Manually download and save as Selected_Document.txt

### Slow embedding generation
**Normal:** First run downloads models (~200MB) and generates embeddings
- Sentence transformer model: ~100MB
- Cross-encoder model: ~80MB
- Subsequent runs are much faster (models cached)

### Out of memory errors
**Solution:** Reduce chunk count by increasing chunk_size:
```python
chunk_size = 1000  # or even 2000
```

## File Structure

```
retrieval/
├── text_extractor.py              # Document extraction script
├── RAG_app.py                     # Main RAG application
├── run_experiments.py             # Experiment guide script
├── generate_deepdive_questions.py # Deep-dive Q&A generator
├── requirements.txt               # Python dependencies
├── Selected_Document.txt          # Extracted document (87KB)
├── .env                          # Your API keys (DO NOT COMMIT)
├── .env.template                 # Template for .env
├── .gitignore                    # Git ignore rules
├── README.md                     # Full documentation
├── SETUP_GUIDE.md                # This file
└── .venv/                        # Virtual environment (created)
```

## Next Steps

1. **Add Your API Key** to `.env` (most important!)
2. **Run RAG_app.py** to test the system
3. **Ask Test Questions** and document answers
4. **Run Experiments** with different chunk sizes
5. **Update README.md** with your findings
6. **Explore** different documents or Wikipedia articles

## Advanced Usage

### Using a Different Document

#### From Wikipedia:
Edit `text_extractor.py`, line 60:
```python
url = "https://en.wikipedia.org/wiki/Machine_learning"  # or any article
```
Run: `python text_extractor.py`

#### From a PDF:
1. Add PDF extraction function to `text_extractor.py` (template provided in assignment)
2. Place PDF in project root
3. Update `main()` to call PDF extractor

### Customizing RAG Parameters

Edit `RAG_app.py`:
```python
# Line 34-35: Chunk configuration
chunk_size = 500        # Characters per chunk
chunk_overlap = 50      # Overlapping characters

# Line 36: Embedding model
model_name = "sentence-transformers/all-distilroberta-v1"
# Alternatives: all-MiniLM-L6-v2 (faster, smaller), all-mpnet-base-v2 (larger, more accurate)

# Line 37: Retrieval count
top_k = 20             # Initial retrieval count

# Line 40: Re-ranking model  
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Line 41: Final context count
top_m = 8              # Chunks sent to ChatGPT

# Line 222: ChatGPT model
model="gpt-4"          # Or "gpt-3.5-turbo" for faster/cheaper

# Line 226: Temperature
temperature=0.0        # 0=deterministic, 1=creative

# Line 227: Answer length
max_tokens=500         # Maximum answer length
```

### Saving FAISS Index for Faster Startup

Add to `RAG_app.py` after building index:
```python
import faiss
# After line 75 (faiss_index.add...)
faiss.write_index(faiss_index, "document_index.faiss")

# To load on subsequent runs:
# faiss_index = faiss.read_index("document_index.faiss")
```

## Tips for Best Results

1. **Ask Specific Questions:** "What is X?" works better than "Tell me about X"
2. **Stay Within Document Scope:** Questions about content not in the document will return "I don't know"
3. **Chunk Size Guidelines:**
   - Technical docs: 300-500 characters
   - Narrative text: 500-800 characters
   - Legal documents: 800-1200 characters
4. **Overlap Guidelines:** 10-20% of chunk_size is optimal
5. **Re-ranking:** Always keep top_k > top_m (e.g., 20 vs 8) for best results

## Resources

- [Sentence Transformers Docs](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify your Python version: `python --version` (should be 3.8+)
3. Verify dependencies: `pip list`
4. Check `.env` file has valid API key
5. Ensure `Selected_Document.txt` exists and has content

---

**Ready to start? Run:** `python RAG_app.py`
