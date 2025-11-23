"""
run_experiments.py - Automated experiments for RAG system evaluation
"""

import sys
import os

# Questions to test the RAG system
TEST_QUESTIONS = [
    "What is artificial intelligence and when was it founded?",
    "How do neural networks work in AI?",
    "What are the ethical concerns related to AI?",
]

# Parameter configurations to test
CONFIGS = [
    {"chunk_size": 500, "chunk_overlap": 50, "name": "Default"},
    {"chunk_size": 1000, "chunk_overlap": 100, "name": "Large Chunks"},
    {"chunk_size": 250, "chunk_overlap": 25, "name": "Small Chunks"},
]

def run_experiment(config, questions):
    """Run experiment with given configuration"""
    print(f"\n{'='*70}")
    print(f"Configuration: {config['name']}")
    print(f"chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
    print(f"{'='*70}\n")
    
    # This is a placeholder - you would modify RAG_app.py to use these parameters
    # and then import and run it, or use subprocess to run it separately
    
    print("To run this experiment:")
    print(f"1. Edit RAG_app.py: set chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
    print(f"2. Run RAG_app.py and ask the test questions")
    print(f"3. Document the answers in README.md\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("Answer: [Run RAG_app.py to get answer]\n")


if __name__ == "__main__":
    print("RAG System Experiment Guide")
    print("="*70)
    print("\nTest Questions:")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"{i}. {q}")
    
    print("\n\nExperiment Configurations:")
    for config in CONFIGS:
        run_experiment(config, TEST_QUESTIONS)
    
    print("\n" + "="*70)
    print("IMPORTANT: Before running experiments, make sure to:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Run each configuration and document results in README.md")
    print("="*70)
