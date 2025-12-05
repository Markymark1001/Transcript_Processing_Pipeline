#!/usr/bin/env python3
"""
Test script for the transcript processing pipeline
"""

import os
import sys
import json
from pathlib import Path
import config
from text_processor import TextProcessor
from embedding_generator import EmbeddingGenerator
from transcript_processor import TranscriptProcessor

def test_text_processor():
    """Test the text processor functionality."""
    print("Testing TextProcessor...")
    
    processor = TextProcessor()
    
    # Test text cleaning
    test_text = """
    [00:00:00] Interviewer: Good morning, thank you for joining us today.
    [00:00:05] Speaker: Um, thank you for having me. I'm, like, really excited to discuss our findings.
    [00:00:12] Interviewer: Can you tell us about the key discoveries?
    [00:00:18] Speaker: Absolutely. We discovered that machine learning models can significantly improve prediction accuracy.
    """
    
    cleaned = processor.clean_text(test_text)
    print(f"Original length: {len(test_text)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Cleaned text: {cleaned[:200]}...")
    
    # Test statement extraction
    statements = processor.extract_statements(cleaned)
    print(f"Extracted {len(statements)} statements")
    
    for i, stmt in enumerate(statements[:3]):
        print(f"Statement {i+1}: {stmt['text'][:100]}...")
        print(f"  Importance score: {stmt['importance_score']:.3f}")
        print(f"  Entities: {[e['text'] for e in stmt['entities']]}")
    
    print("✓ TextProcessor test passed\n")
    return True

def test_embedding_generator():
    """Test the embedding generator functionality."""
    print("Testing EmbeddingGenerator...")
    
    try:
        generator = EmbeddingGenerator()
        
        test_texts = [
            "This is a test sentence for embedding generation.",
            "Machine learning models can improve prediction accuracy.",
            "The research team discovered important findings."
        ]
        
        embeddings = generator.get_embeddings(test_texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Test similarity search
        query_embedding = embeddings[0]
        similarities = generator.similarity_search(query_embedding, embeddings[1:], top_k=2)
        print(f"Similarity search results: {similarities}")
        
        print("✓ EmbeddingGenerator test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ EmbeddingGenerator test failed: {e}\n")
        return False

def test_transcript_processor():
    """Test the full transcript processor."""
    print("Testing TranscriptProcessor...")
    
    processor = TranscriptProcessor()
    
    # Test with sample transcript
    test_transcript = """
    Meeting Minutes - Product Development Team
    
    [10:00 AM] Team Lead: Welcome everyone. Let's start with the sprint review.
    [10:02 AM] Developer 1: I completed the user authentication module. All tests are passing.
    [10:05 AM] Developer 2: The payment integration is 80% complete. We're facing some API issues.
    [10:08 AM] Team Lead: What's the timeline for resolving those issues?
    [10:10 AM] Developer 2: I estimate another 2-3 days for proper error handling.
    [10:15 AM] Designer: The new UI mockups are ready for review.
    [10:18 AM] Product Manager: Excellent work everyone. The client expects a demo by Friday.
    [10:22 AM] Team Lead: Yes, we should deliver everything by Thursday evening.
    """
    
    result = processor.process_single_transcript(test_transcript, "test_meeting")
    
    print(f"Transcript ID: {result['transcript_id']}")
    print(f"Original length: {result['original_length']}")
    print(f"Cleaned length: {result['cleaned_length']}")
    print(f"Statements extracted: {result['statement_count']}")
    print(f"Entities found: {result['entity_count']}")
    
    if config.INCLUDE_EMBEDDINGS and "transcript_embedding" in result:
        print(f"Embedding dimension: {result['embedding_dim']}")
    
    print("✓ TranscriptProcessor test passed\n")
    return True

def test_sample_creation():
    """Test sample transcript creation."""
    print("Testing sample transcript creation...")
    
    # Import main to use the create_sample_transcripts function
    import main
    
    # Create samples
    main.create_sample_transcripts()
    
    # Check if files were created
    sample_dir = config.TRANSCRIPTS_DIR
    sample_files = list(sample_dir.glob("*.txt"))
    
    print(f"Created {len(sample_files)} sample files:")
    for file_path in sample_files:
        print(f"  - {file_path.name}")
    
    # Verify files have content
    for file_path in sample_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) < 100:
                print(f"✗ Sample file {file_path.name} seems too short")
                return False
    
    print("✓ Sample creation test passed\n")
    return True

def test_batch_processing():
    """Test batch processing with sample files."""
    print("Testing batch processing...")
    
    # First create samples
    import main
    main.create_sample_transcripts()
    
    # Test batch processing
    processor = TranscriptProcessor()
    results = processor.process_transcripts_from_directory()
    
    print(f"Processed {len(results)} transcripts")
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Generate report
    report = processor.generate_summary_report(results)
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Total statements: {report['statement_statistics']['total_statements']}")
    
    print("✓ Batch processing test passed\n")
    return True

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TRANSCRIPT PROCESSING PIPELINE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_text_processor,
        test_embedding_generator,
        test_transcript_processor,
        test_sample_creation,
        test_batch_processing
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Place your transcript files in the data/transcripts/ directory")
        print("2. Run: python main.py")
        print("3. Or for large batches: python batch_processor.py --input-dir /path/to/transcripts")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)