"""
Run RAG system tests and save results
"""
import json
import time
from RAG_app import answer_question

questions = [
    'What is artificial intelligence and when was it founded?',
    'How do neural networks work in AI?',
    'What are the ethical concerns related to AI?'
]

print('\n' + '='*70)
print('Testing RAG System - SMALL CHUNKS Configuration (250/25)')
print('='*70 + '\n')

results = []
for i, question in enumerate(questions, 1):
    print(f'Question {i}: {question}')
    start = time.time()
    answer = answer_question(question)
    elapsed = time.time() - start
    
    results.append({
        'question': question,
        'answer': answer,
        'time_seconds': round(elapsed, 2)
    })
    
    print(f'Answer ({elapsed:.1f}s):\n{answer}\n')
    print('-'*70 + '\n')

# Save results
with open('results_small.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print('='*70)
print('✓ Results saved to results_small.json')
print('='*70)
