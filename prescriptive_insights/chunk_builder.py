"""
Chunk Builder for Prescriptive Insights

Creates rolling window chunks from processed transcript statements,
with support for incremental updates and manifest tracking.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from embedding_generator import EmbeddingGenerator
from .topic_registry import TopicRegistry

class ChunkBuilder:
    """Builds and manages chunks from processed transcript data."""
    
    def __init__(self, 
                 window_size: int = None,
                 overlap: int = None,
                 max_chars: int = None,
                 use_embeddings: bool = True):
        """Initialize chunk builder with configuration."""
        self.window_size = window_size or config.INSIGHTS_CHUNK_WINDOW_SIZE
        self.overlap = overlap or config.INSIGHTS_CHUNK_OVERLAP
        self.max_chars = max_chars or config.INSIGHTS_CHUNK_MAX_CHARS
        self.use_embeddings = use_embeddings
        
        # Initialize components
        self.topic_registry = TopicRegistry()
        self.embedding_generator = None
        
        if self.use_embeddings:
            try:
                self.embedding_generator = EmbeddingGenerator()
            except Exception as e:
                print(f"Warning: Could not initialize embedding generator: {e}")
                self.use_embeddings = False
        
        # Ensure output directory exists
        config.INSIGHTS_CHUNK_DIR.mkdir(exist_ok=True)
        
        # Manifest file path
        self.manifest_path = config.INSIGHTS_CHUNK_DIR / "insights_manifest.json"
        
    def build_chunks(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build chunks from all processed transcripts."""
        print("Starting chunk building process...")
        
        # Load existing manifest
        manifest = self._load_manifest()
        
        # Get all processed transcript files
        transcript_files = list(config.PROCESSED_TRANSCRIPTS_DIR.glob("*_processed.json"))
        print(f"Found {len(transcript_files)} processed transcript files")
        
        # Track statistics
        stats = {
            "total_transcripts": len(transcript_files),
            "processed_transcripts": 0,
            "skipped_transcripts": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        # Process each transcript
        all_chunks = []
        
        for transcript_file in tqdm(transcript_files, desc="Processing transcripts"):
            try:
                # Check if transcript needs processing
                transcript_id = transcript_file.stem.replace("_processed", "")
                file_hash = self._get_file_hash(transcript_file)
                file_mtime = transcript_file.stat().st_mtime
                
                should_process = force_rebuild or self._should_process_transcript(
                    transcript_id, file_hash, file_mtime, manifest
                )
                
                if not should_process:
                    stats["skipped_transcripts"] += 1
                    continue
                
                # Load and process transcript
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Create chunks
                chunks = self._create_chunks(transcript_data, transcript_id)
                
                # Save chunks for this transcript
                self._save_transcript_chunks(transcript_id, chunks)
                
                # Update manifest
                manifest[transcript_id] = {
                    "file_hash": file_hash,
                    "file_mtime": file_mtime,
                    "chunk_count": len(chunks),
                    "last_processed": pd.Timestamp.now().isoformat()
                }
                
                all_chunks.extend(chunks)
                stats["processed_transcripts"] += 1
                stats["total_chunks"] += len(chunks)
                
            except Exception as e:
                error_msg = f"Error processing {transcript_file}: {str(e)}"
                print(error_msg)
                stats["errors"].append(error_msg)
        
        # Save master dataframe
        if all_chunks:
            self._save_master_dataframe(all_chunks)
        
        # Save manifest
        self._save_manifest(manifest)
        
        print(f"Chunk building complete. Processed {stats['processed_transcripts']} transcripts, "
              f"created {stats['total_chunks']} chunks.")
        
        return stats
    
    def _create_chunks(self, transcript_data: Dict[str, Any], transcript_id: str) -> List[Dict[str, Any]]:
        """Create rolling window chunks from transcript statements."""
        statements = transcript_data.get("statements", [])
        if not statements:
            return []
        
        chunks = []
        
        # Sort statements by importance score
        statements.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        
        # Create rolling windows
        for i in range(0, len(statements), self.window_size - self.overlap):
            window_statements = statements[i:i + self.window_size]
            
            if len(window_statements) < 2:  # Skip very small chunks
                continue
            
            # Create chunk
            chunk = self._create_chunk_from_statements(
                window_statements, transcript_id, i
            )
            
            # Check character limit
            if len(chunk["text"]) <= self.max_chars:
                chunks.append(chunk)
            else:
                # Try to split chunk if too long
                sub_chunks = self._split_long_chunk(chunk, window_statements, transcript_id, i)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _create_chunk_from_statements(self, 
                                   statements: List[Dict[str, Any]], 
                                   transcript_id: str, 
                                   chunk_index: int) -> Dict[str, Any]:
        """Create a chunk from a list of statements."""
        # Combine statement texts
        chunk_text = " ".join(stmt["text"] for stmt in statements)
        
        # Calculate combined importance score
        importance_scores = [stmt.get("importance_score", 0) for stmt in statements]
        avg_importance = np.mean(importance_scores) if importance_scores else 0
        
        # Collect all entities
        all_entities = []
        for stmt in statements:
            all_entities.extend(stmt.get("entities", []))
        
        # Find matching topics
        matching_topics = self.topic_registry.find_matching_topics(chunk_text, all_entities)
        
        # Create chunk metadata
        chunk = {
            "chunk_id": f"{transcript_id}_chunk_{chunk_index}",
            "transcript_id": transcript_id,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "statement_count": len(statements),
            "character_count": len(chunk_text),
            "avg_importance_score": float(avg_importance),
            "max_importance_score": float(max(importance_scores)) if importance_scores else 0,
            "entities": all_entities,
            "matching_topics": matching_topics,
            "statement_indices": [stmt.get("start_char", 0) for stmt in statements],
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # Add embeddings if available
        if self.use_embeddings and self.embedding_generator:
            try:
                embedding = self.embedding_generator.get_transcript_embedding(chunk_text)
                if embedding is not None:
                    chunk["embedding"] = embedding.tolist()
                    chunk["embedding_dim"] = len(embedding)
            except Exception as e:
                print(f"Warning: Could not generate embedding for chunk {chunk['chunk_id']}: {e}")
        
        return chunk
    
    def _split_long_chunk(self, 
                        original_chunk: Dict[str, Any], 
                        statements: List[Dict[str, Any]], 
                        transcript_id: str, 
                        chunk_index: int) -> List[Dict[str, Any]]:
        """Split a chunk that exceeds the character limit."""
        sub_chunks = []
        
        # Split statements into smaller groups
        for i in range(0, len(statements), 2):  # Split into pairs
            sub_statements = statements[i:i + 2]
            if not sub_statements:
                continue
            
            sub_chunk = self._create_chunk_from_statements(
                sub_statements, transcript_id, f"{chunk_index}_{i//2}"
            )
            
            if len(sub_chunk["text"]) <= self.max_chars:
                sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _save_transcript_chunks(self, transcript_id: str, chunks: List[Dict[str, Any]]):
        """Save chunks for a specific transcript as JSONL."""
        output_file = config.INSIGHTS_CHUNK_DIR / f"{transcript_id}_chunks.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    def _save_master_dataframe(self, all_chunks: List[Dict[str, Any]]):
        """Save all chunks as a master parquet file."""
        # Convert to DataFrame
        df = pd.DataFrame(all_chunks)
        
        # Fix data types for parquet compatibility
        if "chunk_index" in df.columns:
            df["chunk_index"] = df["chunk_index"].astype(str)
        if "statement_indices" in df.columns:
            df["statement_indices"] = df["statement_indices"].astype(str)
        if "embedding_dim" in df.columns:
            df["embedding_dim"] = df["embedding_dim"].astype("Int64")
        
        # Convert lists to strings for parquet compatibility
        if "matching_topics" in df.columns:
            df["matching_topics"] = df["matching_topics"].apply(
                lambda topics: ",".join(topics) if isinstance(topics, list) else topics
            )
        
        # Save as parquet
        output_file = config.INSIGHTS_CHUNK_DIR / "master_chunks.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"Saved master dataframe with {len(df)} chunks to {output_file}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file contents."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _should_process_transcript(self, 
                                transcript_id: str, 
                                file_hash: str, 
                                file_mtime: float, 
                                manifest: Dict[str, Any]) -> bool:
        """Check if a transcript needs to be processed."""
        if transcript_id not in manifest:
            return True
        
        manifest_entry = manifest[transcript_id]
        
        # Check if file has changed
        if manifest_entry.get("file_hash") != file_hash:
            return True
        
        # Check if file is newer
        if manifest_entry.get("file_mtime", 0) < file_mtime:
            return True
        
        return False
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
        
        return {}
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save manifest to file."""
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

def main():
    """CLI entry point for chunk builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build chunks from processed transcripts")
    parser.add_argument("--rebuild", action="store_true", 
                       help="Force rebuild all chunks")
    parser.add_argument("--window-size", type=int, 
                       default=config.INSIGHTS_CHUNK_WINDOW_SIZE,
                       help="Window size for chunking")
    parser.add_argument("--overlap", type=int, 
                       default=config.INSIGHTS_CHUNK_OVERLAP,
                       help="Overlap between chunks")
    parser.add_argument("--max-chars", type=int, 
                       default=config.INSIGHTS_CHUNK_MAX_CHARS,
                       help="Maximum characters per chunk")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Disable embedding generation")
    
    args = parser.parse_args()
    
    # Create chunk builder
    builder = ChunkBuilder(
        window_size=args.window_size,
        overlap=args.overlap,
        max_chars=args.max_chars,
        use_embeddings=not args.no_embeddings
    )
    
    # Build chunks
    stats = builder.build_chunks(force_rebuild=args.rebuild)
    
    # Print summary
    print("\n=== Chunk Building Summary ===")
    print(f"Total transcripts: {stats['total_transcripts']}")
    print(f"Processed transcripts: {stats['processed_transcripts']}")
    print(f"Skipped transcripts: {stats['skipped_transcripts']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    
    if stats['errors']:
        print(f"Errors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

if __name__ == "__main__":
    main()