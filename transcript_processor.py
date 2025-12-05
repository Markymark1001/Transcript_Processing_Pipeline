import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import config
from text_processor import TextProcessor
from embedding_generator import EmbeddingGenerator

class TranscriptProcessor:
    def __init__(self, hf_token: str = None):
        """Initialize the transcript processor."""
        self.text_processor = TextProcessor()
        self.embedding_generator = None
        self.hf_token = hf_token
        
        # Initialize embedding generator if needed
        if config.INCLUDE_EMBEDDINGS:
            try:
                self.embedding_generator = EmbeddingGenerator()
            except Exception as e:
                print(f"Warning: Could not initialize embedding generator: {e}")
                print("Continuing without embeddings...")
    
    def process_single_transcript(self, transcript_text: str, transcript_id: str = None) -> Dict[str, Any]:
        """Process a single transcript."""
        # Process text with spaCy
        result = self.text_processor.process_transcript(transcript_text, transcript_id)
        
        # Add embeddings if available
        if config.INCLUDE_EMBEDDINGS and self.embedding_generator and "error" not in result:
            # Generate embeddings for the full transcript
            transcript_embedding = self.embedding_generator.get_transcript_embedding(result["cleaned_text"])
            if transcript_embedding is not None:
                result["transcript_embedding"] = transcript_embedding.tolist()
                result["embedding_dim"] = len(transcript_embedding)
            
            # Generate embeddings for statements
            if result["statements"]:
                result["statements"] = self.embedding_generator.get_statement_embeddings(result["statements"])
        
        return result
    
    def process_transcripts_from_directory(self, input_dir: str = None, 
                                         output_file: str = None) -> List[Dict[str, Any]]:
        """Process all transcripts from a directory."""
        input_dir = input_dir or config.TRANSCRIPTS_DIR
        output_file = output_file or config.OUTPUT_DIR / f"processed_transcripts.{config.OUTPUT_FORMAT}"
        
        # Find all transcript files
        file_patterns = ["*.txt", "*.md", "*.transcript", "*.vtt"]
        transcript_files = []
        
        for pattern in file_patterns:
            transcript_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        
        if not transcript_files:
            print(f"No transcript files found in {input_dir}")
            return []
        
        print(f"Found {len(transcript_files)} transcript files")
        
        # Process each transcript
        results = []
        for file_path in tqdm(transcript_files, desc="Processing transcripts"):
            try:
                # Read transcript
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                
                # Generate transcript ID from filename
                transcript_id = Path(file_path).stem
                
                # Process transcript
                result = self.process_single_transcript(transcript_text, transcript_id)
                result["source_file"] = file_path
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({
                    "transcript_id": Path(file_path).stem,
                    "source_file": file_path,
                    "error": str(e)
                })
        
        # Save results
        self._save_results(results, output_file)
        
        return results
    
    def process_transcripts_from_list(self, transcripts: List[Dict[str, str]], 
                                    output_file: str = None) -> List[Dict[str, Any]]:
        """Process transcripts from a list of dictionaries."""
        output_file = output_file or config.OUTPUT_DIR / f"processed_transcripts.{config.OUTPUT_FORMAT}"
        
        results = []
        for transcript_data in tqdm(transcripts, desc="Processing transcripts"):
            try:
                transcript_id = transcript_data.get("id", f"transcript_{len(results)}")
                transcript_text = transcript_data.get("text", "")
                
                result = self.process_single_transcript(transcript_text, transcript_id)
                
                # Add any additional metadata
                for key, value in transcript_data.items():
                    if key not in ["id", "text"]:
                        result[f"metadata_{key}"] = value
                
                results.append(result)
                
            except Exception as e:
                transcript_id = transcript_data.get("id", f"transcript_{len(results)}")
                print(f"Error processing transcript {transcript_id}: {e}")
                results.append({
                    "transcript_id": transcript_id,
                    "error": str(e),
                    **{f"metadata_{k}": v for k, v in transcript_data.items() if k not in ["id", "text"]}
                })
        
        # Save results
        self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save processing results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        if config.OUTPUT_FORMAT == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        elif config.OUTPUT_FORMAT == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif config.OUTPUT_FORMAT == "csv":
            # Flatten nested data for CSV
            flattened_results = []
            for result in results:
                flat_result = {
                    "transcript_id": result.get("transcript_id"),
                    "original_length": result.get("original_length"),
                    "cleaned_length": result.get("cleaned_length"),
                    "cleaned_text": result.get("cleaned_text"),
                    "statement_count": result.get("statement_count"),
                    "entity_count": result.get("entity_count"),
                    "error": result.get("error"),
                    "source_file": result.get("source_file")
                }
                
                # Add embedding info if available
                if "transcript_embedding" in result:
                    flat_result["has_embedding"] = True
                    flat_result["embedding_dim"] = result.get("embedding_dim")
                else:
                    flat_result["has_embedding"] = False
                
                # Add top statements as separate columns
                statements = result.get("statements", [])
                for i, stmt in enumerate(statements[:5]):  # Top 5 statements
                    flat_result[f"statement_{i+1}"] = stmt.get("text", "")
                    flat_result[f"statement_{i+1}_importance"] = stmt.get("importance_score", 0)
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
        
        elif config.OUTPUT_FORMAT == "parquet":
            # Similar to CSV but save as parquet
            flattened_results = []
            for result in results:
                flat_result = {
                    "transcript_id": result.get("transcript_id"),
                    "original_length": result.get("original_length"),
                    "cleaned_length": result.get("cleaned_length"),
                    "cleaned_text": result.get("cleaned_text"),
                    "statement_count": result.get("statement_count"),
                    "entity_count": result.get("entity_count"),
                    "error": result.get("error"),
                    "source_file": result.get("source_file")
                }
                
                if "transcript_embedding" in result:
                    flat_result["has_embedding"] = True
                    flat_result["embedding_dim"] = result.get("embedding_dim")
                else:
                    flat_result["has_embedding"] = False
                
                statements = result.get("statements", [])
                for i, stmt in enumerate(statements[:5]):
                    flat_result[f"statement_{i+1}"] = stmt.get("text", "")
                    flat_result[f"statement_{i+1}_importance"] = stmt.get("importance_score", 0)
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_parquet(output_path, index=False)
        
        print(f"Results saved to {output_path}")
    
    def upload_to_huggingface(self, results: List[Dict[str, Any]], repo_id: str = None):
        """Upload processed results to Hugging Face Hub."""
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi, create_repo
            
            repo_id = repo_id or config.HF_REPO_NAME
            
            # Create dataset
            df = pd.DataFrame(results)
            dataset = Dataset.from_pandas(df)
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=repo_id,
                    token=self.hf_token,
                    private=config.HF_PRIVATE,
                    repo_type="dataset"
                )
                print(f"Created new repository: {repo_id}")
            except Exception:
                print(f"Repository {repo_id} already exists or couldn't be created")
            
            # Push to hub
            dataset.push_to_hub(
                repo_id=repo_id,
                token=self.hf_token,
                private=config.HF_PRIVATE
            )
            
            print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
            
        except ImportError:
            print("Hugging Face datasets library not installed. Install with: pip install datasets")
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report of processing results."""
        total_transcripts = len(results)
        successful_transcripts = len([r for r in results if "error" not in r])
        failed_transcripts = total_transcripts - successful_transcripts
        
        if successful_transcripts == 0:
            return {
                "total_transcripts": total_transcripts,
                "successful_transcripts": 0,
                "failed_transcripts": failed_transcripts,
                "success_rate": 0.0
            }
        
        # Calculate statistics
        total_original_chars = sum(r.get("original_length", 0) for r in results if "error" not in r)
        total_cleaned_chars = sum(r.get("cleaned_length", 0) for r in results if "error" not in r)
        total_statements = sum(r.get("statement_count", 0) for r in results if "error" not in r)
        total_entities = sum(r.get("entity_count", 0) for r in results if "error" not in r)
        
        avg_original_length = total_original_chars / successful_transcripts
        avg_cleaned_length = total_cleaned_chars / successful_transcripts
        avg_statements_per_transcript = total_statements / successful_transcripts
        avg_entities_per_transcript = total_entities / successful_transcripts
        
        # Top entities across all transcripts
        all_entities = []
        for result in results:
            if "error" not in result:
                for stmt in result.get("statements", []):
                    for entity in stmt.get("entities", []):
                        all_entities.append(entity["label"])
        
        entity_counts = pd.Series(all_entities).value_counts().to_dict() if all_entities else {}
        
        report = {
            "total_transcripts": total_transcripts,
            "successful_transcripts": successful_transcripts,
            "failed_transcripts": failed_transcripts,
            "success_rate": successful_transcripts / total_transcripts,
            "text_statistics": {
                "total_original_characters": total_original_chars,
                "total_cleaned_characters": total_cleaned_chars,
                "average_original_length": avg_original_length,
                "average_cleaned_length": avg_cleaned_length,
                "compression_ratio": avg_cleaned_length / avg_original_length if avg_original_length > 0 else 0
            },
            "statement_statistics": {
                "total_statements": total_statements,
                "average_statements_per_transcript": avg_statements_per_transcript,
                "total_entities": total_entities,
                "average_entities_per_transcript": avg_entities_per_transcript
            },
            "top_entity_types": entity_counts
        }
        
        return report