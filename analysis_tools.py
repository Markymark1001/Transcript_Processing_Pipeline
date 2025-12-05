#!/usr/bin/env python3
"""
Tools for analyzing your processed transcript data
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import seaborn as sns

class TranscriptAnalyzer:
    def __init__(self, jsonl_file):
        """Load processed transcript data"""
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} processed transcripts")
    
    def get_summary_stats(self):
        """Get basic statistics"""
        total_statements = sum(item.get('statement_count', 0) for item in self.data)
        total_entities = sum(item.get('entity_count', 0) for item in self.data)
        
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Total transcripts: {len(self.data)}")
        print(f"   Total statements: {total_statements}")
        print(f"   Total entities: {total_entities}")
        print(f"   Avg statements per transcript: {total_statements/len(self.data):.1f}")
        print(f"   Avg entities per transcript: {total_entities/len(self.data):.1f}")
    
    def analyze_entities(self):
        """Analyze named entities"""
        entity_counts = Counter()
        entity_types = Counter()
        
        for item in self.data:
            for statement in item.get('statements', []):
                for entity in statement.get('entities', []):
                    entity_counts[entity['text']] += 1
                    entity_types[entity['label']] += 1
        
        print(f"\n🏷️  TOP ENTITIES:")
        print("   Top 20 mentioned entities:")
        for entity, count in entity_counts.most_common(20):
            print(f"   {entity}: {count} times")
        
        print(f"\n   Entity types:")
        for entity_type, count in entity_types.most_common():
            print(f"   {entity_type}: {count}")
        
        return entity_counts, entity_types
    
    def analyze_statements(self):
        """Analyze statement importance and content"""
        all_statements = []
        importance_scores = []
        
        for item in self.data:
            for statement in item.get('statements', []):
                all_statements.append(statement['text'])
                importance_scores.append(statement['importance_score'])
        
        print(f"\n💬 STATEMENT ANALYSIS:")
        print(f"   Total statements: {len(all_statements)}")
        print(f"   Average importance score: {np.mean(importance_scores):.3f}")
        print(f"   Most important statements (score > 0.8):")
        
        high_importance = []
        for item in self.data:
            for statement in item.get('statements', []):
                if statement['importance_score'] > 0.8:
                    high_importance.append((statement['text'], statement['importance_score']))
        
        for text, score in sorted(high_importance, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   Score {score:.3f}: {text[:100]}...")
        
        return all_statements, importance_scores
    
    def cluster_transcripts(self, n_clusters=5):
        """Cluster transcripts by content similarity"""
        print(f"\n🔍 CLUSTERING ANALYSIS (finding {n_clusters} topic groups):")
        
        # Extract transcript texts
        texts = [item['cleaned_text'] for item in self.data]
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        features = vectorizer.fit_transform(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Analyze clusters
        cluster_info = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_info[cluster_id].append(i)
        
        print(f"   Found {n_clusters} topic clusters:")
        for cluster_id, transcript_indices in cluster_info.items():
            print(f"   Cluster {cluster_id}: {len(transcript_indices)} transcripts")
            
            # Get top words for this cluster
            cluster_features = features[transcript_indices]
            mean_features = np.mean(cluster_features.toarray(), axis=0)
            
            # Get top words
            top_indices = np.argsort(mean_features)[-5:][::-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
            print(f"     Key words: {', '.join(top_words)}")
        
        return clusters, vectorizer
    
    def similarity_search(self, query_text, top_k=5):
        """Find transcripts most similar to a query"""
        print(f"\n🔎 SIMILARITY SEARCH for: '{query_text}'")
        
        # Get embeddings (if available)
        embeddings = []
        transcript_ids = []
        
        for item in self.data:
            if 'transcript_embedding' in item:
                embeddings.append(item['transcript_embedding'])
                transcript_ids.append(item['transcript_id'])
        
        if not embeddings:
            print("   No embeddings found in data. Run processing with embeddings enabled.")
            return
        
        embeddings = np.array(embeddings)
        
        # Simple similarity (you could use more sophisticated methods)
        from sklearn.metrics.pairwise import cosine_similarity
        
        # For demo, use first transcript as "query"
        query_embedding = embeddings[0].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top similar (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        print(f"   Top {top_k} most similar transcripts:")
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i}. {transcript_ids[idx]} (similarity: {similarities[idx]:.3f})")
        
        return similarities
    
    def create_visualizations(self):
        """Create simple visualizations"""
        print(f"\n📈 CREATING VISUALIZATIONS...")
        
        try:
            # 1. Entity type distribution
            entity_types = Counter()
            for item in self.data:
                for statement in item.get('statements', []):
                    for entity in statement.get('entities', []):
                        entity_types[entity['label']] += 1
            
            plt.figure(figsize=(12, 8))
            
            # Entity types pie chart
            plt.subplot(2, 2, 1)
            top_entities = dict(entity_types.most_common(8))
            plt.pie(top_entities.values(), labels=top_entities.keys(), autopct='%1.1f%%')
            plt.title('Top Entity Types')
            
            # Statement importance distribution
            plt.subplot(2, 2, 2)
            importance_scores = []
            for item in self.data:
                for statement in item.get('statements', []):
                    importance_scores.append(statement['importance_score'])
            
            plt.hist(importance_scores, bins=20, alpha=0.7)
            plt.title('Statement Importance Scores')
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
            
            # Transcript length distribution
            plt.subplot(2, 2, 3)
            lengths = [item['cleaned_length'] for item in self.data]
            plt.hist(lengths, bins=20, alpha=0.7)
            plt.title('Transcript Lengths')
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
            
            # Statements per transcript
            plt.subplot(2, 2, 4)
            statement_counts = [item.get('statement_count', 0) for item in self.data]
            plt.hist(statement_counts, bins=20, alpha=0.7)
            plt.title('Statements per Transcript')
            plt.xlabel('Number of Statements')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('output/analysis_visualizations.png', dpi=150, bbox_inches='tight')
            print("   Saved visualizations to: output/analysis_visualizations.png")
            
        except Exception as e:
            print(f"   Error creating visualizations: {e}")
            print("   Install matplotlib and seaborn: pip install matplotlib seaborn wordcloud")
    
    def export_summary_csv(self):
        """Export summary data to CSV for further analysis"""
        summary_data = []
        for item in self.data:
            summary_data.append({
                'transcript_id': item['transcript_id'],
                'original_length': item.get('original_length', 0),
                'cleaned_length': item.get('cleaned_length', 0),
                'statement_count': item.get('statement_count', 0),
                'entity_count': item.get('entity_count', 0),
                'avg_importance': np.mean([s['importance_score'] for s in item.get('statements', [])]) if item.get('statements') else 0
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv('output/transcript_summary.csv', index=False)
        print("   Exported summary to: output/transcript_summary.csv")
        
        return df

def main():
    """Run analysis on your processed data"""
    print("🔍 TRANSCRIPT ANALYSIS TOOLS")
    print("=" * 50)
    
    # Load your processed data
    analyzer = TranscriptAnalyzer('output/drboz_results.jsonl')
    
    # Run different analyses
    analyzer.get_summary_stats()
    entity_counts, entity_types = analyzer.analyze_entities()
    statements, importance_scores = analyzer.analyze_statements()
    
    # More advanced analyses
    clusters, vectorizer = analyzer.cluster_transcripts(n_clusters=5)
    similarities = analyzer.similarity_search("machine learning", top_k=5)
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Export for further analysis
    summary_df = analyzer.export_summary_csv()
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"   Check output/ folder for:")
    print(f"   - analysis_visualizations.png")
    print(f"   - transcript_summary.csv")
    print(f"   - drboz_results.jsonl (your original processed data)")

if __name__ == "__main__":
    main()