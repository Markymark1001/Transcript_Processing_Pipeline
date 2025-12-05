#!/usr/bin/env python3
"""
Batch Processing Script for Large Numbers of Transcripts

This script is optimized for processing 500+ transcripts efficiently with
memory management, progress tracking, and error handling.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import config
from transcript_processor import TranscriptProcessor

class BatchProcessor:
    def __init__(self, hf_token: str = None, max_workers: int = 4):
        """Initialize the batch processor."""
        self.hf_token = hf_token
        self.max_workers = max_workers
        self.processor = TranscriptProcessor(hf_token=hf_token)
        
        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_processing_time": 0,
            "errors": []
        }
    
    def find_transcript_files(self, input_dir: str) -> List[str]:
        """Find all transcript files in the input directory and subdirectories."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Supported file extensions
        extensions = ["*.txt", "*.md", "*.transcript", "*.vtt"]
        transcript_files = []
        
        for ext in extensions:
            transcript_files.extend(input_path.rglob(ext))
        
        self.stats["total_files"] = len(transcript_files)
        print(f"Found {len(transcript_files)} transcript files")
        
        return [str(f) for f in transcript_files]
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single transcript file."""
        start_time = time.time()
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            # Generate transcript ID from file path
            transcript_id = Path(file_path).stem
            
            # Process the transcript
            result = self.processor.process_single_transcript(transcript_text, transcript_id)
            result["source_file"] = file_path
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats["errors"].append(error_msg)
            
            return {
                "transcript_id": Path(file_path).stem,
                "source_file": file_path,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_batch_sequential(self, file_paths: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """Process files sequentially (memory efficient)."""
        results = []
        output_file = output_file or config.OUTPUT_DIR / f"batch_processed.{config.OUTPUT_FORMAT}"
        
        # Create output directory
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Process files with progress bar
        for file_path in tqdm(file_paths, desc="Processing transcripts"):
            result = self.process_single_file(file_path)
            results.append(result)
            
            # Update statistics
            if "error" in result:
                self.stats["failed_files"] += 1
            else:
                self.stats["processed_files"] += 1
            
            # Save intermediate results periodically
            if len(results) % 50 == 0:
                self._save_intermediate_results(results, output_file)
        
        # Save final results
        self._save_final_results(results, output_file)
        
        return results
    
    def process_batch_parallel(self, file_paths: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """Process files in parallel (faster but uses more memory)."""
        results = []
        output_file = output_file or config.OUTPUT_DIR / f"batch_processed_parallel.{config.OUTPUT_FORMAT}"
        
        # Create output directory
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                             for file_path in file_paths}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_file), total=len(file_paths), 
                              desc="Processing transcripts"):
                result = future.result()
                results.append(result)
                
                # Update statistics
                if "error" in result:
                    self.stats["failed_files"] += 1
                else:
                    self.stats["processed_files"] += 1
                
                # Save intermediate results periodically
                if len(results) % 50 == 0:
                    self._save_intermediate_results(results, output_file)
        
        # Save final results
        self._save_final_results(results, output_file)
        
        return results
    
    def process_batch_streaming(self, file_paths: List[str], output_file: str = None) -> Iterator[Dict[str, Any]]:
        """Process files and yield results one by one (memory efficient for very large datasets)."""
        output_file = output_file or config.OUTPUT_DIR / f"batch_streaming.{config.OUTPUT_FORMAT}"
        
        # Create output directory
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Open output file for streaming
        if config.OUTPUT_FORMAT == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for file_path in tqdm(file_paths, desc="Processing transcripts"):
                    result = self.process_single_file(file_path)
                    
                    # Update statistics
                    if "error" in result:
                        self.stats["failed_files"] += 1
                    else:
                        self.stats["processed_files"] += 1
                    
                    # Write to file immediately
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written immediately
                    
                    yield result
        else:
            # For other formats, collect in memory
            results = []
            for file_path in tqdm(file_paths, desc="Processing transcripts"):
                result = self.process_single_file(file_path)
                results.append(result)
                
                # Update statistics
                if "error" in result:
                    self.stats["failed_files"] += 1
                else:
                    self.stats["processed_files"] += 1
                
                yield result
            
            # Save final results for non-JSONL formats
            self._save_final_results(results, output_file)
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save intermediate results to prevent data loss."""
        if config.OUTPUT_FORMAT == "jsonl":
            intermediate_file = output_file.replace(".jsonl", "_intermediate.jsonl")
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _save_final_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save final results using the processor's save method."""
        self.processor._save_results(results, output_file)
    
    def generate_batch_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive batch processing report."""
        # Get basic statistics from processor
        basic_report = self.processor.generate_summary_report(results)
        
        # Add batch-specific statistics
        batch_report = {
            **basic_report,
            "batch_statistics": {
                "total_files_found": self.stats["total_files"],
                "processing_method": "sequential" if self.max_workers == 1 else f"parallel ({self.max_workers} workers)",
                "total_errors": len(self.stats["errors"]),
                "error_rate": len(self.stats["errors"]) / max(self.stats["total_files"], 1),
                "average_processing_time_per_file": sum(
                    r.get("processing_time", 0) for r in results
                ) / max(len(results), 1)
            },
            "errors": self.stats["errors"][:10]  # First 10 errors
        }
        
        return batch_report
    
    def validate_transcripts(self, file_paths: List[str]) -> Dict[str, Any]:
        """Validate transcript files before processing."""
        validation_results = {
            "total_files": len(file_paths),
            "valid_files": 0,
            "invalid_files": 0,
            "empty_files": 0,
            "very_large_files": 0,
            "file_size_distribution": {},
            "issues": []
        }
        
        file_sizes = []
        
        for file_path in file_paths:
            try:
                # Check file size
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)
                
                # Check if file is empty
                if file_size == 0:
                    validation_results["empty_files"] += 1
                    validation_results["issues"].append(f"Empty file: {file_path}")
                    continue
                
                # Check if file is too large
                if file_size > 10 * 1024 * 1024:  # 10MB
                    validation_results["very_large_files"] += 1
                    validation_results["issues"].append(f"Very large file: {file_path} ({file_size/1024/1024:.1f}MB)")
                
                # Try to read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    
                    if not content.strip():
                        validation_results["empty_files"] += 1
                        validation_results["issues"].append(f"File with no content: {file_path}")
                    else:
                        validation_results["valid_files"] += 1
                
            except Exception as e:
                validation_results["invalid_files"] += 1
                validation_results["issues"].append(f"Cannot read {file_path}: {str(e)}")
        
        # Calculate file size distribution
        if file_sizes:
            file_sizes.sort()
            n = len(file_sizes)
            validation_results["file_size_distribution"] = {
                "min_size": file_sizes[0],
                "max_size": file_sizes[-1],
                "median_size": file_sizes[n//2],
                "avg_size": sum(file_sizes) / n
            }
        
        return validation_results

def main():
    parser = argparse.ArgumentParser(description="Batch process large numbers of transcripts")
    parser.add_argument("--input-dir", type=str, required=True, 
                       help="Directory containing transcript files")
    parser.add_argument("--output-file", type=str, 
                       help="Output file path")
    parser.add_argument("--method", choices=["sequential", "parallel", "streaming"], 
                       default="sequential", help="Processing method")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum number of parallel workers")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate files, don't process them")
    parser.add_argument("--hf-token", type=str, 
                       help="Hugging Face API token")
    parser.add_argument("--upload-to-hf", action="store_true", 
                       help="Upload results to Hugging Face")
    parser.add_argument("--hf-repo", type=str, 
                       help="Hugging Face repository name")
    
    args = parser.parse_args()
    
    # Initialize batch processor
    processor = BatchProcessor(hf_token=args.hf_token, max_workers=args.max_workers)
    
    # Find transcript files
    print("Finding transcript files...")
    file_paths = processor.find_transcript_files(args.input_dir)
    
    if not file_paths:
        print("No transcript files found!")
        return
    
    # Validate files
    print("Validating transcript files...")
    validation = processor.validate_transcripts(file_paths)
    print(f"Validation complete: {validation['valid_files']} valid, {validation['invalid_files']} invalid")
    
    if validation["issues"]:
        print("Issues found:")
        for issue in validation["issues"][:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(validation["issues"]) > 10:
            print(f"  ... and {len(validation['issues']) - 10} more issues")
    
    if args.validate_only:
        print("Validation complete. Exiting.")
        return
    
    # Process files
    print(f"Processing {len(file_paths)} files using {args.method} method...")
    start_time = time.time()
    
    if args.method == "sequential":
        results = processor.process_batch_sequential(file_paths, args.output_file)
    elif args.method == "parallel":
        results = processor.process_batch_parallel(file_paths, args.output_file)
    elif args.method == "streaming":
        results = list(processor.process_batch_streaming(file_paths, args.output_file))
    
    processing_time = time.time() - start_time
    processor.stats["total_processing_time"] = processing_time
    
    # Generate report
    print("\nGenerating batch report...")
    report = processor.generate_batch_report(results)
    
    # Save report
    report_file = config.OUTPUT_DIR / "batch_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nBatch processing complete!")
    print(f"Total files: {report['batch_statistics']['total_files_found']}")
    print(f"Successfully processed: {report['successful_transcripts']}")
    print(f"Failed: {report['batch_statistics']['total_errors']}")
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per file: {report['batch_statistics']['average_processing_time_per_file']:.2f} seconds")
    print(f"Results saved to: {args.output_file or config.OUTPUT_DIR}")
    print(f"Report saved to: {report_file}")
    
    # Upload to Hugging Face if requested
    if args.upload_to_hf:
        print("\nUploading to Hugging Face...")
        processor.processor.upload_to_huggingface(results, repo_id=args.hf_repo)

if __name__ == "__main__":
    main()