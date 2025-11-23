# Chunk Size Experiment Results Template

## Purpose
This template helps you systematically document your chunk size experiments.
Copy your actual results here after running experiments, then transfer insights to README.md.

## Test Questions
1. What is artificial intelligence and when was it founded?
2. How do neural networks work in AI?
3. What are the ethical concerns related to AI?

---

## Experiment 1: Default Configuration (500/50)
**Parameters:** chunk_size=500, chunk_overlap=50
**Total chunks:** ~176

### Question 1: What is artificial intelligence and when was it founded?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Observations:**
- Completeness: [Was the answer complete?]
- Accuracy: [Was the information correct?]
- Context: [Did it have enough context?]
- Response time: [How fast? ~seconds]

### Question 2: How do neural networks work in AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Observations:**
- [Your notes]

### Question 3: What are the ethical concerns related to AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Observations:**
- [Your notes]

### Overall Assessment
**Strengths:**
- [What worked well with this configuration?]

**Weaknesses:**
- [What didn't work as well?]

**Best for:**
- [What types of questions did this handle best?]

---

## Experiment 2: Large Chunks (1000/100)
**Parameters:** chunk_size=1000, chunk_overlap=100
**Total chunks:** ~88

### Question 1: What is artificial intelligence and when was it founded?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- Better/Worse/Same: [Your assessment]
- Why: [Reasoning]

### Question 2: How do neural networks work in AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- [Your notes]

### Question 3: What are the ethical concerns related to AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- [Your notes]

### Overall Assessment
**Strengths:**
- More context per chunk
- [Other strengths]

**Weaknesses:**
- Less precise retrieval?
- [Other weaknesses]

**Best for:**
- [What types of questions?]

---

## Experiment 3: Small Chunks (250/25)
**Parameters:** chunk_size=250, chunk_overlap=25
**Total chunks:** ~352

### Question 1: What is artificial intelligence and when was it founded?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- Better/Worse/Same: [Your assessment]
- Why: [Reasoning]

### Question 2: How do neural networks work in AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- [Your notes]

### Question 3: What are the ethical concerns related to AI?
**Answer:**
```
[Paste answer here]
```
**Quality Rating:** [1-5] ⭐⭐⭐⭐⭐
**Comparison to Default:**
- [Your notes]

### Overall Assessment
**Strengths:**
- More precise retrieval?
- [Other strengths]

**Weaknesses:**
- Less context per chunk
- [Other weaknesses]

**Best for:**
- [What types of questions?]

---

## Cross-Experiment Analysis

### Answer Quality Comparison
| Question | Default (500) | Large (1000) | Small (250) | Winner |
|----------|---------------|--------------|-------------|--------|
| Q1: AI definition | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | [?] |
| Q2: Neural networks | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | [?] |
| Q3: Ethics | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | [?] |

### Key Findings

#### Impact of Chunk Size
**Observation 1: [What you learned]**
- Example: "Larger chunks provided more comprehensive answers for broad questions"

**Observation 2: [What you learned]**
- Example: "Smaller chunks were better for specific factual lookups"

**Observation 3: [What you learned]**
- Example: "Default size (500) balanced precision and context well"

#### Impact of Overlap
**Observation 1: [What you learned]**
- Example: "10% overlap (50/500) was sufficient to preserve sentence continuity"

**Observation 2: [What you learned]**
- Example: "Noticed fewer 'broken thoughts' compared to no overlap"

#### Trade-offs Observed

**Speed vs. Quality:**
- Large chunks: [Faster/Slower?] retrieval, [Better/Worse?] quality
- Small chunks: [Faster/Slower?] retrieval, [Better/Worse?] quality

**Context vs. Precision:**
- Large chunks: More context, [more/less] precise?
- Small chunks: Less context, [more/less] precise?

**Memory vs. Performance:**
- More chunks = more memory but [better/worse] retrieval?

### Question Type Analysis

**Factual Questions (e.g., "When was AI founded?"):**
- Best chunk size: [500/1000/250]
- Reasoning: [Why?]

**Conceptual Questions (e.g., "How do neural networks work?"):**
- Best chunk size: [500/1000/250]
- Reasoning: [Why?]

**Broad Questions (e.g., "What are ethical concerns?"):**
- Best chunk size: [500/1000/250]
- Reasoning: [Why?]

---

## Recommendations

### For This Document Type (Technical Wikipedia Article)
**Optimal Configuration:**
- chunk_size: [Your recommendation]
- chunk_overlap: [Your recommendation]
- Reasoning: [Why these values?]

### For Other Document Types

**Narrative Text (Stories, News):**
- Recommended chunk_size: [Value]
- Recommended chunk_overlap: [Value]
- Reasoning: [Why?]

**Code Documentation:**
- Recommended chunk_size: [Value]
- Recommended chunk_overlap: [Value]
- Reasoning: [Why?]

**Legal/Dense Documents:**
- Recommended chunk_size: [Value]
- Recommended chunk_overlap: [Value]
- Reasoning: [Why?]

### General Guidelines

**Chunk Size:**
- Too small (< 200): [Problem observed]
- Optimal (200-800): [Benefits observed]
- Too large (> 1500): [Problem observed]

**Overlap:**
- Too small (< 5%): [Problem observed]
- Optimal (10-20%): [Benefits observed]
- Too large (> 30%): [Problem observed]

---

## Unexpected Findings

### Surprises
1. [Something unexpected you discovered]
2. [Another surprise]

### Counterintuitive Results
1. [Something that didn't work as expected]
2. [Why it might have happened]

---

## Future Experiments to Try

Based on your results, what would you test next?

1. [Experiment idea]
   - Parameters: [What to change]
   - Expected outcome: [Hypothesis]

2. [Experiment idea]
   - Parameters: [What to change]
   - Expected outcome: [Hypothesis]

---

## Notes for README.md

### Best Quotes/Insights to Include:
1. "[Insight from your experiments]"
2. "[Another key finding]"

### Graphs/Tables to Create:
- [ ] Answer quality by chunk size
- [ ] Response time comparison
- [ ] Chunk count vs. accuracy

### Key Takeaways (for conclusion):
1. [Main lesson learned]
2. [Secondary insight]
3. [Practical recommendation]

---

## Appendix: Retrieved Chunks Analysis

### Example: Question 1, Default Config
**Retrieved chunks (top 8 after re-ranking):**
1. [First few words of chunk 1...]
2. [First few words of chunk 2...]
...

**Relevance:**
- Highly relevant: [X chunks]
- Somewhat relevant: [X chunks]
- Not relevant: [X chunks]

**Coverage:**
- Did chunks contain all needed information? [Yes/No]
- Were there gaps? [Describe]

---

**Remember to:**
- ✅ Run each configuration
- ✅ Ask all 3 questions per configuration
- ✅ Document answers and observations
- ✅ Compare across configurations
- ✅ Draw conclusions and make recommendations
- ✅ Transfer insights to README.md
