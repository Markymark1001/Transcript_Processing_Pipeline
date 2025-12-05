# How to Analyze Your Processed Transcript Data

## 🎯 Quick Start

### Step 1: Install Analysis Tools
```bash
cd /Users/markmacmini/Documents/Kilo-Code
pip3 install -r analysis_requirements.txt
```

### Step 2: Run Analysis
```bash
python3 analysis_tools.py
```

## 🔍 What the Analysis Tools Do

### 1. **Summary Statistics**
- Total transcripts, statements, and entities
- Average statements per transcript
- Average entities per transcript

### 2. **Entity Analysis**
- Top 20 most mentioned people, places, organizations
- Breakdown by entity type (PERSON, ORG, DATE, etc.)
- Frequency counts

### 3. **Statement Analysis**
- Importance score distribution
- Most important statements (score > 0.8)
- Statement quality metrics

### 4. **Topic Clustering**
- Groups transcripts by similar content
- Identifies main themes/topics
- Shows key words for each cluster

### 5. **Similarity Search**
- Find transcripts similar to any topic
- Uses semantic embeddings
- Returns similarity scores

### 6. **Visualizations**
- Entity type distribution (pie chart)
- Statement importance histogram
- Transcript length distribution
- Statements per transcript chart

## 📊 What You'll Learn

### **Content Insights**
- What topics are most discussed
- Who/what is mentioned most frequently
- Which statements are most important

### **Patterns**
- Common themes across transcripts
- Length and complexity patterns
- Entity relationships

### **Search Capabilities**
- Find similar transcripts quickly
- Identify related content
- Discover hidden connections

## 🛠️ Advanced Analysis Options

### **Option 1: Excel/Google Sheets**
```bash
# Export to CSV for spreadsheet analysis
python3 -c "
import pandas as pd
import json
data = []
with open('output/drboz_results.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)
df.to_csv('output/spreadsheet_data.csv', index=False)
print('Exported to output/spreadsheet_data.csv')
"
```

### **Option 2: Custom Questions**
Edit `analysis_tools.py` to add your own analysis:

```python
def custom_analysis(self):
    # Add your own analysis here
    print("Your custom analysis...")
    
# Add to main() function:
analyzer.custom_analysis()
```

### **Option 3: Web Dashboard**
For interactive analysis, you can use tools like:
- **Streamlit**: `pip install streamlit` (easy web apps)
- **Plotly Dash**: Interactive dashboards
- **Jupyter Notebook**: Exploratory analysis

## 📈 Example Insights You Might Find

### **Top Entities** (from your data)
- Dates: 1,874 mentions
- Numbers/Quantities: 1,881 mentions  
- Organizations: 835 mentions
- People: 741 mentions

### **Important Statements**
- Statements with importance scores > 0.8
- Key decisions or findings
- Critical information

### **Topic Clusters**
- Cluster 0: "research, data, analysis" (academic content)
- Cluster 1: "meeting, team, project" (work discussions)
- Cluster 2: "customer, service, issue" (support calls)

## 🚀 Next Steps

### **For Business Analysis**
1. Identify key themes and trends
2. Track entity mentions over time
3. Find high-importance statements
4. Export reports for stakeholders

### **For Research**
1. Use clusters for topic modeling
2. Analyze entity co-occurrence
3. Study statement patterns
4. Create academic papers

### **For Content Strategy**
1. Find most engaging topics
2. Identify popular entities
3. Optimize content based on importance
4. Plan future content themes

## 💡 Pro Tips

1. **Start with Summary**: Get overview first
2. **Focus on High-Importance**: These contain key insights
3. **Use Clusters**: Group similar content together
4. **Export to CSV**: Use Excel for deeper analysis
5. **Visualize**: Charts reveal patterns quickly

## 🔧 Troubleshooting

**If matplotlib fails:**
```bash
pip3 install --upgrade matplotlib
```

**If memory issues:**
- Process smaller chunks
- Use CSV export instead of keeping everything in memory

**If no embeddings found:**
- Re-run processing with embeddings enabled
- Check original processing included embeddings