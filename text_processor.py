import re
import spacy
from typing import List, Dict, Any, Tuple
from collections import Counter
import config

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with spaCy model."""
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            print(f"Loaded spaCy model: {config.SPACY_MODEL}")
        except OSError:
            print(f"spaCy model {config.SPACY_MODEL} not found. Downloading...")
            from spacy.cli import download
            download(config.SPACY_MODEL)
            self.nlp = spacy.load(config.SPACY_MODEL)
        
        # Add custom pipeline components if needed
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize transcript text."""
        if not text:
            return ""
        
        # Remove timestamps (common formats: [00:00:00], 00:00:00, etc.)
        text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
        
        # Remove speaker labels (common formats: Speaker:, [Speaker], etc.)
        text = re.sub(r'\[?[A-Za-z]+(?:\s+[A-Za-z]+)*\]?:', '', text)
        
        # Remove extra whitespace
        if config.NORMALIZE_WHITESPACE:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove filler words
        if config.REMOVE_FILLER_WORDS:
            filler_pattern = r'\b(?:' + '|'.join(map(re.escape, config.FILLER_WORDS)) + r')\b'
            text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()  # Clean up after filler removal
        
        # Remove repetitions (simple approach)
        if config.REMOVE_REPETITIONS:
            text = self._remove_repetitions(text)
        
        return text
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repeated words or phrases."""
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            # Skip if same as previous word (case-insensitive)
            if prev_word and word.lower() == prev_word.lower():
                continue
            cleaned_words.append(word)
            prev_word = word
        
        return ' '.join(cleaned_words)
    
    def extract_statements(self, text: str) -> List[Dict[str, Any]]:
        """Extract important statements from cleaned text."""
        if not text:
            return []
        
        # Process with spaCy
        doc = self.nlp(text)
        statements = []
        
        for sent in doc.sents:
            # Skip very short sentences
            if len(sent.text.strip()) < config.MIN_STATEMENT_LENGTH:
                continue
            
            # Extract statement information
            statement_info = {
                "text": sent.text.strip(),
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "length": len(sent.text),
                "tokens": len(sent),
                "entities": [],
                "sentiment": None,
                "importance_score": 0.0
            }
            
            # Extract named entities
            for ent in sent.ents:
                statement_info["entities"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char - sent.start_char,
                    "end": ent.end_char - sent.start_char
                })
            
            # Calculate importance score based on various factors
            importance = self._calculate_importance(sent, statement_info)
            statement_info["importance_score"] = importance
            
            # Calculate sentiment if possible
            try:
                from textblob import TextBlob
                blob = TextBlob(sent.text)
                statement_info["sentiment"] = {
                    "polarity": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity
                }
            except ImportError:
                pass  # textblob not available, skip sentiment
            
            statements.append(statement_info)
        
        # Sort by importance score and limit
        statements.sort(key=lambda x: x["importance_score"], reverse=True)
        return statements[:config.MAX_STATEMENTS_PER_TRANSCRIPT]
    
    def _calculate_importance(self, sent, statement_info: Dict[str, Any]) -> float:
        """Calculate importance score for a statement."""
        score = 0.0
        
        # Length factor (prefer medium-length sentences)
        length = len(sent)
        if 20 <= length <= 100:
            score += 0.3
        elif 100 < length <= 200:
            score += 0.2
        
        # Entity factor (sentences with entities are more important)
        if statement_info["entities"]:
            score += 0.2 * min(len(statement_info["entities"]) / 3, 1.0)
        
        # Part-of-speech factors
        has_verb = any(token.pos_ == "VERB" for token in sent)
        has_noun = any(token.pos_ == "NOUN" or token.pos_ == "PROPN" for token in sent)
        
        if has_verb and has_noun:
            score += 0.2
        
        # Question/exclamation factor
        if sent.text.strip().endswith('?'):
            score += 0.1
        elif sent.text.strip().endswith('!'):
            score += 0.1
        
        # Modal verbs and adjectives (indicate opinions/important statements)
        has_modal = any(token.tag_ == "MD" for token in sent)  # MD = modal verb
        has_adj = any(token.pos_ == "ADJ" for token in sent)
        
        if has_modal:
            score += 0.1
        if has_adj:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def process_transcript(self, transcript_text: str, transcript_id: str = None) -> Dict[str, Any]:
        """Process a single transcript and return structured data."""
        if not transcript_text:
            return {"error": "Empty transcript"}
        
        # Truncate if too long
        if len(transcript_text) > config.MAX_TRANSCRIPT_LENGTH:
            transcript_text = transcript_text[:config.MAX_TRANSCRIPT_LENGTH]
        
        # Clean the text
        cleaned_text = self.clean_text(transcript_text)
        
        # Extract important statements
        statements = self.extract_statements(cleaned_text)
        
        # Prepare result
        result = {
            "transcript_id": transcript_id,
            "original_length": len(transcript_text),
            "cleaned_length": len(cleaned_text),
            "cleaned_text": cleaned_text,
            "statements": statements,
            "statement_count": len(statements),
            "entity_count": sum(len(s["entities"]) for s in statements),
            "processing_info": {
                "spacy_model": config.SPACY_MODEL,
                "config": {
                    "remove_filler_words": config.REMOVE_FILLER_WORDS,
                    "normalize_whitespace": config.NORMALIZE_WHITESPACE,
                    "min_statement_length": config.MIN_STATEMENT_LENGTH,
                    "max_statements": config.MAX_STATEMENTS_PER_TRANSCRIPT
                }
            }
        }
        
        return result