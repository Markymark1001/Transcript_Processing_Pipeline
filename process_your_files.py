#!/usr/bin/env python3
"""
Script to process your transcript files from documents/subtitles-DrBoz
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_processor import BatchProcessor

def main():
    # Path to your transcript files
    input_dir = "/Users/markmacmini/Documents/subtitles-DrBoz"
    
    # Output file
    output_file = "output/drboz_processed_transcripts.jsonl"
    
    print(f"Processing transcripts from: {input_dir}")
    print(f"Output will be saved to: {output_file}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist!")
        print("Please check the path and make sure the directory exists.")
        return
    
    # Initialize batch processor
    processor = BatchProcessor()
    
    try:
        # Process files
        results = processor.process_batch_sequential(input_dir, output_file)
        
        # Generate report
        report = processor.generate_batch_report(results)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total files processed: {report['batch_statistics']['total_files_found']}")
        print(f"Successful: {report['successful_transcripts']}")
        print(f"Failed: {report['batch_statistics']['total_errors']}")
        print(f"Success rate: {report['success_rate']:.2%}")
        print(f"Total statements extracted: {report['statement_statistics']['total_statements']}")
        print(f"Total entities found: {report['statement_statistics']['total_entities']}")
        print(f"Results saved to: {output_file}")
        
        if report['batch_statistics']['total_errors'] > 0:
            print(f"\nErrors encountered:")
            for error in report['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(report['errors']) > 5:
                print(f"  ... and {len(report['errors']) - 5} more errors")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()