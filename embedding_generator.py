import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import config

class EmbeddingGenerator:
    def __init__(self, model_name: str = None):
        """Initialize the embedding generator with Hugging Face model."""
        self.model_name = model_name or config.HF_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Hugging Face model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def get_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        batch_size = batch_size or config.BATCH_SIZE
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    
    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """Process a single batch of texts."""
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # Use mean pooling of the last hidden state
            embeddings = self._mean_pooling(
                model_output.last_hidden_state, 
                encoded_input['attention_mask']
            )
        
        # Move to CPU and convert to numpy
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to the last hidden state."""
        # Expand attention_mask to match hidden_state size
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum embeddings and divide by attention mask sum
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def get_statement_embeddings(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for extracted statements."""
        if not statements:
            return []
        
        # Extract statement texts
        statement_texts = [stmt["text"] for stmt in statements]
        
        # Generate embeddings
        embeddings = self.get_embeddings(statement_texts)
        
        # Add embeddings to statements
        statements_with_embeddings = []
        for i, stmt in enumerate(statements):
            stmt_copy = stmt.copy()
            if i < len(embeddings):
                stmt_copy["embedding"] = embeddings[i].tolist()
                stmt_copy["embedding_dim"] = len(embeddings[i])
            statements_with_embeddings.append(stmt_copy)
        
        return statements_with_embeddings
    
    def get_transcript_embedding(self, transcript_text: str) -> Optional[np.ndarray]:
        """Generate a single embedding for an entire transcript."""
        if not transcript_text:
            return None
        
        embeddings = self.get_embeddings([transcript_text])
        return embeddings[0] if len(embeddings) > 0 else None
    
    def similarity_search(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar embeddings to a query."""
        if len(candidate_embeddings) == 0:
            return []
        
        # Calculate cosine similarity
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "embedding": candidate_embeddings[idx].tolist()
            })
        
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        np.save(filepath, embeddings)
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        embeddings = np.load(filepath)
        print(f"Loaded embeddings from {filepath}")
        return embeddings