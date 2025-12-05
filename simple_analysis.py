#!/usr/bin/env python3
"""
Simple analysis tool using only built-in Python libraries
"""

import json
import csv
from collections import Counter, defaultdict

def analyze_transcripts():
    """Analyze your processed transcript data with basic Python"""
    print("🔍 ANALYZING YOUR TRANSCRIPT DATA")
    print("=" * 50)
    
    # Load your processed data
    data = []
    try:
        with open('output/drboz_results.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"✅ Loaded {len(data)} processed transcripts")
    except FileNotFoundError:
        print("❌ Error: output/drboz_results.jsonl not found!")
        print("   Make sure you ran the processing first.")
        return
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total transcripts: {len(data)}")
    
    total_statements = sum(item.get('statement_count', 0) for item in data)
    total_entities = sum(item.get('entity_count', 0) for item in data)
    total_chars = sum(item.get('cleaned_length', 0) for item in data)
    
    print(f"   Total statements: {total_statements}")
    print(f"   Total entities: {total_entities}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg statements per transcript: {total_statements/len(data):.1f}")
    print(f"   Avg entities per transcript: {total_entities/len(data):.1f}")
    print(f"   Avg transcript length: {total_chars/len(data):.0f} characters")
    
    # Entity analysis
    print(f"\n🏷️  ENTITY ANALYSIS:")
    entity_counts = Counter()
    entity_types = Counter()
    
    for item in data:
        for statement in item.get('statements', []):
            for entity in statement.get('entities', []):
                entity_counts[entity['text']] += 1
                entity_types[entity['label']] += 1
    
    print(f"   Top 15 most mentioned entities:")
    for entity, count in entity_counts.most_common(15):
        print(f"   {entity}: {count} times")
    
    print(f"\n   Entity types found:")
    for entity_type, count in entity_types.most_common():
        print(f"   {entity_type}: {count}")
    
    # Statement importance analysis
    print(f"\n💬 STATEMENT IMPORTANCE:")
    importance_scores = []
    high_importance = []
    
    for item in data:
        for statement in item.get('statements', []):
            importance_scores.append(statement['importance_score'])
            if statement['importance_score'] > 0.8:
                high_importance.append(statement['text'])
    
    print(f"   Average importance score: {sum(importance_scores)/len(importance_scores):.3f}")
    print(f"   High importance statements (>0.8): {len(high_importance)}")
    
    print(f"\n   Top 10 most important statements:")
    all_statements = []
    for item in data:
        for statement in item.get('statements', []):
            all_statements.append((statement['text'], statement['importance_score']))
    
    # Sort by importance and show top 10
    all_statements.sort(key=lambda x: x[1], reverse=True)
    for i, (text, score) in enumerate(all_statements[:10], 1):
        print(f"   {i}. ({score:.3f}) {text[:80]}...")
    
    # Transcript length analysis
    print(f"\n📏 TRANSCRIPT LENGTH ANALYSIS:")
    lengths = [item.get('cleaned_length', 0) for item in data]
    lengths.sort()
    
    print(f"   Shortest transcript: {min(lengths):,} characters")
    print(f"   Longest transcript: {max(lengths):,} characters")
    print(f"   Median length: {lengths[len(lengths)//2]:,} characters")
    print(f"   Average length: {sum(lengths)/len(lengths):.0f} characters")
    
    # Create simple CSV export
    print(f"\n💾 EXPORTING TO CSV:")
    try:
        with open('output/simple_summary.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['transcript_id', 'length', 'statements', 'entities', 'avg_importance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                statements = item.get('statements', [])
                avg_importance = sum(s['importance_score'] for s in statements) / len(statements) if statements else 0
                
                writer.writerow({
                    'transcript_id': item.get('transcript_id', ''),
                    'length': item.get('cleaned_length', 0),
                    'statements': len(statements),
                    'entities': item.get('entity_count', 0),
                    'avg_importance': round(avg_importance, 3)
                })
        
        print("   ✅ Exported summary to: output/simple_summary.csv")
        print("   📊 Open this file in Excel or Google Sheets for charts!")
        
    except Exception as e:
        print(f"   ❌ Error exporting CSV: {e}")
    
    # Find interesting patterns
    print(f"\n🔍 INTERESTING PATTERNS:")
    
    # Find transcripts with most entities
    max_entities = max(data, key=lambda x: x.get('entity_count', 0))
    print(f"   Transcript with most entities: {max_entities.get('transcript_id', 'unknown')} ({max_entities.get('entity_count', 0)} entities)")
    
    # Find transcripts with most statements
    max_statements = max(data, key=lambda x: x.get('statement_count', 0))
    print(f"   Transcript with most statements: {max_statements.get('transcript_id', 'unknown')} ({max_statements.get('statement_count', 0)} statements)")
    
    # Find longest and shortest transcripts
    longest = max(data, key=lambda x: x.get('cleaned_length', 0))
    shortest = min(data, key=lambda x: x.get('cleaned_length', 0))
    print(f"   Longest transcript: {longest.get('transcript_id', 'unknown')} ({longest.get('cleaned_length', 0):,} chars)")
    print(f"   Shortest transcript: {shortest.get('transcript_id', 'unknown')} ({shortest.get('cleaned_length', 0):,} chars)")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"   Check output/simple_summary.csv for spreadsheet-ready data")
    print(f"   Your most important statements are listed above")
    print(f"   Top entities show what's mentioned most frequently")

if __name__ == "__main__":
    analyze_transcripts()