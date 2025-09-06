#!/usr/bin/env python3
"""
Simple test for text cleaning without heavy dependencies
"""

import re

def basic_clean(text: str) -> str:
    """Basic text cleaning"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
    
    return text.strip()

def preprocess_for_embeddings(text: str) -> str:
    """Preprocess text specifically for embedding generation"""
    if not text or len(text.strip()) < 3:
        return ""
    
    # Light cleaning - preserve semantic meaning
    text = basic_clean(text)
    
    # If text is too short after cleaning, return original
    if len(text.strip()) < 3:
        return text.strip()
    
    # Simple sentence extraction
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if sentences:
        meaningful_sentences = [s for s in sentences if len(s.split()) > 2]
        if meaningful_sentences:
            return ' '.join(meaningful_sentences)
    
    # If no meaningful sentences found, return cleaned text
    return text.strip()

def test_text_cleaning():
    """Test the text cleaning functions"""
    test_cases = [
        "What is machine learning?",
        "How to learn Python programming?",
        "Best career advice for students",
        "What are the benefits of artificial intelligence?",
        "How does quantum computing work?"
    ]
    
    print("Testing text cleaning functions:")
    print("=" * 50)
    
    for i, test_text in enumerate(test_cases, 1):
        cleaned = preprocess_for_embeddings(test_text)
        print(f"Test {i}:")
        print(f"  Original: '{test_text}'")
        print(f"  Cleaned:  '{cleaned}'")
        print(f"  Length:   {len(cleaned)}")
        print()

if __name__ == "__main__":
    test_text_cleaning()
