#!/usr/bin/env python3
"""
Transcript Processing Pipeline

This script processes transcript files using spaCy for text cleaning and statement extraction,
and Hugging Face models for embeddings generation.

Usage:
    python main.py --input-dir ./data/transcripts --output-file ./output/processed.jsonl
    python main.py --sample --test-embedding
    python main.py --upload-to-hf --hf-token YOUR_TOKEN
"""

import argparse
import json
import os
from pathlib import Path
import config
from transcript_processor import TranscriptProcessor

def create_sample_transcripts():
    """Create sample transcript files for testing."""
    sample_dir = config.TRANSCRIPTS_DIR
    sample_dir.mkdir(exist_ok=True)
    
    # Sample transcript 1
    transcript1 = """[00:00:00] Interviewer: Good morning, thank you for joining us today.
    [00:00:05] Speaker: Thank you for having me. I'm excited to discuss our latest research findings.
    [00:00:12] Interviewer: Can you tell us about the key discoveries your team made?
    [00:00:18] Speaker: Absolutely. We discovered that machine learning models can significantly improve prediction accuracy by up to 40% compared to traditional methods. This is a groundbreaking finding that could revolutionize how we approach data analysis.
    [00:00:35] Interviewer: That's impressive. What are the practical applications?
    [00:00:40] Speaker: The applications are numerous. We're looking at healthcare diagnostics, financial forecasting, and even climate modeling. The potential impact is enormous.
    [00:00:55] Interviewer: How long did this research take?
    [00:01:00] Speaker: We've been working on this for about three years now. It involved extensive testing and validation across multiple datasets.
    [00:01:12] Interviewer: What's next for your research?
    [00:01:17] Speaker: We're currently working on making these models more accessible to smaller organizations. We believe democratizing AI is crucial for innovation.
    [00:01:30] Interviewer: Thank you for sharing this with us.
    [00:01:33] Speaker: You're welcome. We're excited about the future."""
    
    # Sample transcript 2
    transcript2 = """Meeting Minutes - Product Development Team
    
    [10:00 AM] Team Lead: Welcome everyone. Let's start with the sprint review.
    [10:02 AM] Developer 1: I completed the user authentication module. All tests are passing and the documentation is updated.
    [10:05 AM] Developer 2: The payment integration is 80% complete. We're facing some issues with the third-party API response times.
    [10:08 AM] Team Lead: What's the timeline for resolving those issues?
    [10:10 AM] Developer 2: I estimate another 2-3 days. We need to implement proper error handling and retry logic.
    [10:15 AM] Designer: The new UI mockups are ready for review. I've incorporated all the feedback from last week's user testing session.
    [10:18 AM] Product Manager: Excellent work everyone. The client is expecting a demo by Friday. Can we meet that deadline?
    [10:22 AM] Team Lead: Yes, we should be able to deliver everything by Thursday evening. This gives us buffer time for final testing.
    [10:25 AM] Developer 1: I'll help Developer 2 with the payment integration to speed things up.
    [10:28 AM] Team Lead: Great teamwork. Let's sync up tomorrow morning to track progress.
    [10:30 AM] All: Agreed."""
    
    # Sample transcript 3
    transcript3 = """Customer Service Call Transcript
    
    Agent: Thank you for calling TechSupport. How can I help you today?
    Customer: Hi, I'm having trouble with my internet connection. It keeps dropping every few minutes.
    Agent: I'm sorry to hear that. Can you tell me when this started happening?
    Customer: It started about two days ago. At first it was occasional, but now it's happening constantly.
    Agent: Have you tried restarting your router?
    Customer: Yes, I've done that multiple times. The issue persists even after restarts.
    Agent: Let me check your account and run some diagnostics. Can I have your customer ID please?
    Customer: It's 12345-67890.
    Agent: Thank you. I can see there have been some connectivity issues in your area. Our technicians are already working on it.
    Customer: When do you expect this to be resolved?
    Agent: We estimate the repairs will be completed within 24 hours. Would you like me to set up automatic updates?
    Customer: Yes, please. That would be helpful.
    Agent: I've added you to the notification list. You'll receive updates via SMS and email.
    Customer: Thank you for your help.
    Agent: You're welcome. Is there anything else I can assist you with today?
    Customer: No, that's all. Thanks again.
    Agent: Have a great day!"""
    
    # Write sample files
    with open(sample_dir / "sample1.txt", "w", encoding="utf-8") as f:
        f.write(transcript1)
    
    with open(sample_dir / "sample2.txt", "w", encoding="utf-8") as f:
        f.write(transcript2)
    
    with open(sample_dir / "sample3.txt", "w", encoding="utf-8") as f:
        f.write(transcript3)
    
    print(f"Created 3 sample transcript files in {sample_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process transcripts with spaCy and Hugging Face")
    parser.add_argument("--input-dir", type=str, help="Directory containing transcript files")
    parser.add_argument("--output-file", type=str, help="Output file path")
    parser.add_argument("--sample", action="store_true", help="Create sample transcripts for testing")
    parser.add_argument("--test-embedding", action="store_true", help="Test embedding generation")
    parser.add_argument("--upload-to-hf", action="store_true", help="Upload results to Hugging Face")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repository name")
    parser.add_argument("--format", type=str, choices=["jsonl", "json", "csv", "parquet"], 
                       help="Output format")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation")
    
    args = parser.parse_args()
    
    # Create sample transcripts if requested
    if args.sample:
        create_sample_transcripts()
        print("Sample transcripts created. You can now run the processing pipeline.")
        return
    
    # Update config based on arguments
    if args.format:
        config.OUTPUT_FORMAT = args.format
    
    if args.no_embeddings:
        config.INCLUDE_EMBEDDINGS = False
    
    # Initialize processor
    print("Initializing transcript processor...")
    processor = TranscriptProcessor(hf_token=args.hf_token)
    
    # Test embedding generation if requested
    if args.test_embedding:
        print("Testing embedding generation...")
        test_text = "This is a test sentence for embedding generation."
        embedding = processor.embedding_generator.get_embeddings([test_text])
        print(f"Generated embedding shape: {embedding.shape}")
        print("Embedding test successful!")
        return
    
    # Process transcripts
    print("Starting transcript processing...")
    if args.input_dir:
        results = processor.process_transcripts_from_directory(
            input_dir=args.input_dir,
            output_file=args.output_file
        )
    else:
        # Use default directory
        results = processor.process_transcripts_from_directory()
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = processor.generate_summary_report(results)
    
    # Save report
    report_file = config.OUTPUT_DIR / "processing_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Processed {report['successful_transcripts']}/{report['total_transcripts']} transcripts successfully")
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Total statements extracted: {report['statement_statistics']['total_statements']}")
    print(f"Total entities found: {report['statement_statistics']['total_entities']}")
    print(f"Results saved to: {args.output_file or config.OUTPUT_DIR / f'processed_transcripts.{config.OUTPUT_FORMAT}'}")
    print(f"Report saved to: {report_file}")
    
    # Upload to Hugging Face if requested
    if args.upload_to_hf:
        print("\nUploading to Hugging Face...")
        processor.upload_to_huggingface(results, repo_id=args.hf_repo)

if __name__ == "__main__":
    main()