import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import spacy
from typing import List, Dict
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class TextCleaner:
    """Handles text cleaning and preprocessing for NLP tasks"""
    
    def __init__(self, use_spacy: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model if available
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. Using NLTK instead.")
                self.logger.info("Install with: python -m spacy download en_core_web_sm")
    
    def basic_clean(self, text: str) -> str:
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
    
    def advanced_clean(self, text: str) -> str:
        """Advanced cleaning with more sophisticated rules"""
        text = self.basic_clean(text)
        
        # Remove common Quora-specific patterns
        quora_patterns = [
            r'originally answered:.*?\n',
            r'continue reading.*?$',
            r'see more.*?$',
            r'upvote.*?\d+',
            r'views.*?\d+',
            r'updated.*?\d+.*?ago'
        ]
        
        for pattern in quora_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove repeated punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!\?]{2,}', '!', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.!\?,:;])', r'\1', text)
        text = re.sub(r'([\.!\?])\s*([a-zA-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize and lemmatize text"""
        if self.nlp:
            return self._spacy_process(text, remove_stopwords)
        else:
            return self._nltk_process(text, remove_stopwords)
    
    def _spacy_process(self, text: str, remove_stopwords: bool) -> List[str]:
        """Process text using spaCy"""
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip punctuation, spaces, and stop words
            if (not token.is_punct and 
                not token.is_space and 
                not token.like_url and
                not token.like_email and
                len(token.text) > 2):
                
                if remove_stopwords and token.is_stop:
                    continue
                
                # Use lemma
                tokens.append(token.lemma_.lower())
        
        return tokens
    
    def _nltk_process(self, text: str, remove_stopwords: bool) -> List[str]:
        """Process text using NLTK"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # POS tagging for better lemmatization
        pos_tags = pos_tag(tokens)
        
        processed_tokens = []
        for word, pos in pos_tags:
            # Skip punctuation and short words
            if word in string.punctuation or len(word) < 3:
                continue
            
            # Skip stop words if requested
            if remove_stopwords and word.lower() in self.stop_words:
                continue
            
            # Convert POS tag to WordNet format
            wordnet_pos = self._get_wordnet_pos(pos)
            
            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(word.lower(), wordnet_pos)
            processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert TreeBank POS tags to WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            sentences = sent_tokenize(text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def preprocess_for_embeddings(self, text: str) -> str:
        """Preprocess text specifically for embedding generation"""
        # Light cleaning - preserve semantic meaning
        text = self.basic_clean(text)
        
        # Remove very short sentences (likely noise)
        sentences = self.extract_sentences(text)
        meaningful_sentences = [s for s in sentences if len(s.split()) > 3]
        
        return ' '.join(meaningful_sentences)