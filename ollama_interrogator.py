#!/usr/bin/env python3
"""
Interrogate your transcript data using local Ollama models
"""

import json
import re
import requests
import time

STOPWORDS = {
    "the", "is", "and", "or", "of", "to", "a", "an", "in", "on", "for",
    "with", "that", "as", "at", "by", "from", "about", "it", "this",
    "these", "those", "which", "be", "was", "were", "are", "but", "if",
    "so", "we", "you", "your", "our", "they", "their"
}

MAX_CONTEXT_KEYWORDS = 5
MIN_KEYWORD_SCORE = 2
WINDOW_PADDING = 60

class OllamaInterrogator:
    def __init__(self, model_name="qwen3:latest", base_url="http://localhost:11434"):
        """Initialize Ollama connection"""
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        
        # Test connection
        try:
            response = self.session.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✅ Connected to Ollama!")
                print(f"   Available models: {', '.join(model_names)}")
                if model_name not in model_names:
                    print(f"   ⚠️  Model '{model_name}' not found. Using first available model.")
                    self.model_name = model_names[0] if model_names else "llama2"
            else:
                print(f"❌ Cannot connect to Ollama at {base_url}")
                print("   Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("   Install Ollama: https://ollama.ai")
    
    def load_transcripts(self, jsonl_file):
        """Load processed transcript data"""
        self.data = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
            print(f"✅ Loaded {len(self.data)} transcripts")
        except FileNotFoundError:
            print(f"❌ Error: {jsonl_file} not found!")
            return False
        return True
    
    def ask_ollama(self, prompt, context=""):
        """Send question to Ollama with context"""
        full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a detailed answer based only on the context provided above. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
"""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response')
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def search_transcripts(self, query_text, limit=5):
        """Search for transcripts matching query"""
        matching_transcripts = []
        
        for item in self.data:
            # Search in cleaned text and statements
            text_to_search = item.get('cleaned_text', '').lower()
            statements_text = ' '.join([s.get('text', '') for s in item.get('statements', [])]).lower()
            
            if query_text.lower() in text_to_search or query_text.lower() in statements_text:
                matching_transcripts.append(item)
                if len(matching_transcripts) >= limit:
                    break
        
        return matching_transcripts
    
    def extract_keywords(self, text):
        """Tokenize user queries and remove stopwords"""
        if not text:
            return []
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        keywords = []
        seen = set()
        for token in tokens:
            if len(token) <= 1 or token in STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                keywords.append(token)
            if len(keywords) >= MAX_CONTEXT_KEYWORDS:
                break
        return keywords
    
    def score_transcripts_by_keywords(self, keywords, limit=5):
        """Score transcripts based on keyword frequency"""
        scored = []
        if not keywords:
            return scored
        for item in self.data:
            cleaned_original = item.get('cleaned_text', '')
            cleaned_lower = cleaned_original.lower()
            statements = [s.get('text', '') for s in item.get('statements', [])]
            statements_lower = [s.lower() for s in statements]
            combined_text = ' '.join([cleaned_lower] + statements_lower)
            score = sum(combined_text.count(keyword) for keyword in keywords)
            statement_matches = []
            for original, lower in zip(statements, statements_lower):
                matched_keywords = [keyword for keyword in keywords if keyword in lower]
                if matched_keywords:
                    statement_matches.append({
                        'text': original,
                        'keywords': matched_keywords
                    })
            text_windows = []
            for keyword in keywords:
                start = cleaned_lower.find(keyword)
                if start == -1:
                    continue
                end = min(len(cleaned_original), start + len(keyword) + WINDOW_PADDING)
                window_text = cleaned_original[max(0, start - WINDOW_PADDING):end].replace('\n', ' ').strip()
                if window_text:
                    text_windows.append({'keyword': keyword, 'window': window_text})
            scored.append({
                'item': item,
                'score': score,
                'statement_matches': statement_matches,
                'text_windows': text_windows
            })
        scored.sort(key=lambda entry: entry['score'], reverse=True)
        return scored[:limit]
    
    def build_context_snippets(self, hits, keywords):
        """Build context snippet that surfaces evidence for each keyword"""
        if not hits:
            return ""
        lines = []
        for entry in hits:
            item = entry['item']
            transcript_id = item.get('transcript_id', 'unknown')
            lines.append(f"Transcript {transcript_id} (score {entry['score']}):")
            displayed_any = False
            for keyword in keywords:
                statements = [match['text'] for match in entry['statement_matches'] if keyword in match['keywords']]
                if statements:
                    lines.append(f"  • '{keyword}' mentioned in:")
                    for stmt in statements[:2]:
                        lines.append(f"      ↳ {stmt}")
                    displayed_any = True
                    continue
                window = next((w['window'] for w in entry['text_windows'] if w['keyword'] == keyword), None)
                if window:
                    lines.append(f"  • '{keyword}' context window: {window}")
                    displayed_any = True
            if not displayed_any:
                preview = item.get('cleaned_text', '').replace('\n', ' ').strip()
                if preview:
                    lines.append(f"  • Transcript preview: {preview[:160]}")
            lines.append("")
        return '\n'.join(lines).strip()
    
    def interactive_mode(self, verbose=False):
        """Start interactive Q&A session with keyword-driven retrieval"""
        print(f"\n🤖 OLLAMA INTERACTIVE MODE")
        print(f"   Model: {self.model_name}")
        print(f"   Transcripts loaded: {len(self.data)}")
        print(f"   Derived keywords will be displayed before searching")
        print(f"   Type 'quit' to exit, 'help' for commands")
        print("=" * 50)
        verbose_mode = verbose
        
        while True:
            try:
                user_input = input(f"\n🔍 Ask about your transcripts: ").strip()
                command = user_input.lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if command == 'help':
                    self.show_help()
                    continue
                
                if command == 'stats':
                    self.show_statistics()
                    continue
                
                if command.startswith('search '):
                    query = user_input[7:]
                    self.search_and_display(query)
                    continue
                
                if command.startswith('summarize '):
                    transcript_id = user_input[10:]
                    self.summarize_transcript(transcript_id)
                    continue
                
                if command in ('verbose on', 'debug on'):
                    verbose_mode = True
                    print("🔍 Verbose/debug logging enabled")
                    continue
                
                if command in ('verbose off', 'debug off'):
                    verbose_mode = False
                    print("🔍 Verbose/debug logging disabled")
                    continue
                
                keywords = self.extract_keywords(user_input)
                keyword_display = ", ".join(keywords) if keywords else "None"
                print(f"🔑 Derived keywords: {keyword_display}")
                if not keywords:
                    print("   Unable to extract meaningful keywords. Try rephrasing or `search <term>`.")
                    continue
                
                hits = self.score_transcripts_by_keywords(keywords, limit=5)
                if not hits:
                    print("⚠️ No transcripts are currently loaded to match against.")
                    continue
                
                best_score = hits[0]['score']
                context_snippet = self.build_context_snippets(hits[:3], keywords)
                if best_score < MIN_KEYWORD_SCORE:
                    print(f"⚠️ No transcript surpassed the keyword overlap threshold ({MIN_KEYWORD_SCORE}).")
                    if verbose_mode:
                        scores = [hit['score'] for hit in hits[:3]]
                        print(f"   🔎 Debug scores for top transcripts: {scores}")
                    if context_snippet:
                        print("\n🔎 Fallback context from best matches:")
                        print(context_snippet)
                    print("   Try `search <term>` or provide more specific wording.")
                    continue
                
                context = "Relevant transcript content:\n\n" + (context_snippet or "No matching statements could be extracted.")
                print("🤔 Thinking...")
                response = self.ask_ollama(user_input, context)
                print(f"\n🤖 Ollama Response:")
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show available commands"""
        print(f"\n📚 AVAILABLE COMMANDS:")
        print(f"   help            - Show this help")
        print(f"   stats           - Show transcript statistics")
        print(f"   search <term>   - Search for transcripts containing <term>")
        print(f"   summarize <id>  - Summarize a specific transcript")
        print(f"   verbose on/off  - Toggle debugging logs for keyword matching")
        print(f"   quit            - Exit the program")
        print(f"\n💡 When asking questions, derived keywords and evidence snippets are shown before Ollama is queried.")
    
    def show_statistics(self):
        """Show basic statistics"""
        total_statements = sum(item.get('statement_count', 0) for item in self.data)
        total_entities = sum(item.get('entity_count', 0) for item in self.data)
        
        print(f"\n📊 TRANSCRIPT STATISTICS:")
        print(f"   Total transcripts: {len(self.data)}")
        print(f"   Total statements: {total_statements}")
        print(f"   Total entities: {total_entities}")
        print(f"   Avg statements/transcript: {total_statements/len(self.data):.1f}")
        print(f"   Avg entities/transcript: {total_entities/len(self.data):.1f}")
    
    def search_and_display(self, query):
        """Search and display results"""
        results = self.search_transcripts(query, limit=10)
        
        print(f"\n🔍 SEARCH RESULTS for '{query}':")
        if results:
            for i, item in enumerate(results, 1):
                print(f"   {i}. {item.get('transcript_id', 'unknown')}")
                print(f"      Statements: {item.get('statement_count', 0)}")
                print(f"      Entities: {item.get('entity_count', 0)}")
                # Show preview
                preview = item.get('cleaned_text', '')[:100]
                print(f"      Preview: {preview}...")
                print()
        else:
            print("   No matching transcripts found.")
    
    def summarize_transcript(self, transcript_id):
        """Summarize a specific transcript"""
        for item in self.data:
            if item.get('transcript_id', '').lower() == transcript_id.lower():
                print(f"\n📝 SUMMARIZING: {transcript_id}")
                
                # Create summary prompt
                statements = item.get('statements', [])
                summary_text = '\n'.join([s.get('text', '') for s in statements[:10]])  # Top 10 statements
                
                prompt = f"Please summarize the following transcript content in 3-4 bullet points:\n\n{summary_text}"
                
                print("🤔 Generating summary...")
                response = self.ask_ollama(prompt)
                
                print(f"\n📋 SUMMARY:")
                print(response)
                return
        
        print(f"❌ Transcript '{transcript_id}' not found.")
        print(f"   Available transcripts: {[item.get('transcript_id', 'unknown') for item in self.data[:10]]}...")
    
    def batch_analysis(self, questions_file=None):
        """Run batch analysis with predefined questions"""
        if not questions_file:
            # Default questions for transcript analysis
            questions = [
                "What are the main topics discussed across these transcripts?",
                "Who are the most frequently mentioned people or organizations?",
                "What are the key decisions or action items mentioned?",
                "What time periods or dates are most referenced?",
                "What are the common problems or challenges discussed?",
                "What solutions or recommendations are provided?",
                "What are the most important statements based on their content?",
                "What trends or patterns emerge across all transcripts?"
            ]
        else:
            # Load questions from file
            try:
                with open(questions_file, 'r') as f:
                    questions = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"❌ Questions file {questions_file} not found.")
                return
        
        print(f"\n🔍 BATCH ANALYSIS - {len(questions)} questions")
        print("=" * 50)
        
        results = {}
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            print("🤔 Processing...")
            
            # Create context from sample of transcripts
            context = "Sample transcript content:\n\n"
            sample_size = min(10, len(self.data))
            for j in range(sample_size):
                item = self.data[j]
                context += f"Transcript {item.get('transcript_id', 'unknown')}:\n"
                statements = item.get('statements', [])[:3]  # Top 3 statements
                for stmt in statements:
                    context += f"- {stmt.get('text', '')}\n"
                context += "\n"
            
            response = self.ask_ollama(question, context)
            results[question] = response
            print(f"✅ Completed")
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(1)
        
        # Save results
        output_file = "output/ollama_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Batch analysis saved to: {output_file}")
        return results

def main():
    """Main function to run Ollama interrogator"""
    print("🤖 OLLAMA TRANSCRIPT INTERROGATOR")
    print("=" * 50)
    
    # Initialize
    interrogator = OllamaInterrogator()
    
    # Load transcripts
    if not interrogator.load_transcripts('output/drboz_results.jsonl'):
        return
    
    print(f"\n🎯 Choose mode:")
    print(f"   1. Interactive Q&A (ask questions about your data)")
    print(f"   2. Batch analysis (answer common analysis questions)")
    print(f"   3. Custom question (ask one specific question)")
    
    try:
        choice = input(f"\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interrogator.interactive_mode()
        elif choice == "2":
            interrogator.batch_analysis()
        elif choice == "3":
            question = input("Enter your question: ").strip()
            if question:
                print("🤔 Processing...")
                # Use sample context
                context = "Sample transcript content:\n\n"
                for i in range(min(5, len(interrogator.data))):
                    item = interrogator.data[i]
                    context += f"Transcript {item.get('transcript_id', 'unknown')}:\n"
                    statements = item.get('statements', [])[:2]
                    for stmt in statements:
                        context += f"- {stmt.get('text', '')}\n"
                    context += "\n"
                
                response = interrogator.ask_ollama(question, context)
                print(f"\n🤖 Response:")
                print(response)
        else:
            print("❌ Invalid choice.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()