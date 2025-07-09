# Text Summarization Tool

A comprehensive Python-based text summarization tool that implements multiple Natural Language Processing (NLP) techniques to automatically generate concise summaries from lengthy articles and documents.

## 🚀 Features

- **Multiple Summarization Methods**: 
  - TF-IDF (Term Frequency-Inverse Document Frequency) with cosine similarity
  - Frequency-based word scoring approach
- **Interactive Mode**: Real-time text input and summarization
- **Comprehensive Statistics**: Compression ratios, word counts, and performance metrics
- **Smart Text Processing**: Automatic sentence tokenization and stop word removal
- **Customizable Summary Length**: Adjustable summary ratio (default: 30% of original)
- **Professional Output**: Clean, formatted summaries with detailed analytics

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```bash
nltk>=3.8
scikit-learn>=1.0
numpy>=1.21
transformers>=4.0 (optional, for advanced features)
torch>=1.9 (optional, for transformer models)
newspaper3k>=0.2 (optional, for web scraping)
beautifulsoup4>=4.10 (optional, for HTML parsing)
requests>=2.25 (optional, for web content)
```

## 🛠️ Installation

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install nltk scikit-learn numpy
   ```
3. **Download NLTK data** (automatic on first run):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## 💻 Usage

### Basic Usage

```bash
python text_summarizer.py
```

### Programmatic Usage

```python
from text_summarizer import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Your text to summarize
text = "Your long article or document text here..."

# Generate summary using TF-IDF method
summary = summarizer.summarize(text, method='tfidf', summary_ratio=0.3)

# Generate summary using frequency method
summary = summarizer.summarize(text, method='frequency', summary_ratio=0.3)

# Get statistics
stats = summarizer.get_text_statistics(text, summary)
print(f"Compression ratio: {stats['compression_ratio']}")
```

## 🔧 Configuration Options

### Summarization Methods

1. **TF-IDF Method** (`method='tfidf'`):
   - Uses Term Frequency-Inverse Document Frequency
   - Calculates cosine similarity between sentences
   - Best for: Technical documents, news articles
   - Recommended for most use cases

2. **Frequency Method** (`method='frequency'`):
   - Uses word frequency scoring
   - Simpler and faster processing
   - Best for: General content, quick summaries

### Summary Length

- **`summary_ratio`**: Controls summary length (0.1 to 0.8)
  - `0.1` = Very short (10% of original)
  - `0.3` = Default (30% of original)
  - `0.5` = Longer summary (50% of original)

## 📊 Output Examples

### Sample Input
```
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents, any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals...
```

### Sample Output
```
TF-IDF METHOD:
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Modern AI techniques are pervasive and are too numerous to list here. High-profile examples of AI include autonomous vehicles, such as drones and self-driving cars, medical diagnosis, creating art, proving mathematical theorems, playing games, search engines, online assistants, image recognition, spam filtering, predicting judicial decisions and targeting online advertisements.

STATISTICS:
Original: 12 sentences, 287 words
Summary: 3 sentences, 89 words
Compression: 69.0%
```

## 🏗️ Code Structure

```
text_summarizer.py
├── TextSummarizer (Main Class)
│   ├── preprocess_text()          # Text cleaning and preprocessing
│   ├── extract_sentences()        # Sentence tokenization
│   ├── calculate_word_frequency()  # Word frequency calculation
│   ├── score_sentences()          # Sentence scoring
│   ├── tfidf_summarize()          # TF-IDF based summarization
│   ├── frequency_summarize()      # Frequency based summarization
│   ├── summarize()                # Main summarization function
│   └── get_text_statistics()      # Statistics calculation
└── main()                         # Interactive demo and examples
```

## 🎯 Algorithm Details

### TF-IDF Approach
1. **Sentence Extraction**: Uses NLTK's sentence tokenizer
2. **Vectorization**: Creates TF-IDF vectors for each sentence
3. **Similarity Calculation**: Computes cosine similarity matrix
4. **Ranking**: Scores sentences based on average similarity
5. **Selection**: Selects top-ranked sentences maintaining original order

### Frequency Approach
1. **Word Frequency**: Calculates normalized word frequencies
2. **Sentence Scoring**: Scores sentences based on word frequency averages
3. **Ranking**: Ranks sentences by cumulative word scores
4. **Selection**: Selects highest-scoring sentences

## 🔍 Performance Metrics

The tool provides comprehensive statistics:

- **Compression Ratio**: Percentage reduction in text length
- **Word Count**: Original vs. summary word counts
- **Sentence Count**: Original vs. summary sentence counts
- **Processing Time**: Time taken for summarization (in advanced mode)

## 🚨 Troubleshooting

### Common Issues

1. **NLTK Data Error**:
   ```bash
   # Manual download
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **Memory Issues with Large Texts**:
   - Break large texts into smaller chunks
   - Use `summary_ratio=0.1` for very long documents

3. **Poor Summary Quality**:
   - Ensure input text has clear sentence structure
   - Try different `summary_ratio` values
   - Switch between TF-IDF and frequency methods

### Installation Issues

If you encounter package installation problems:

```bash
# Try installing packages individually
pip install nltk
pip install scikit-learn
pip install numpy

# Or use conda
conda install nltk scikit-learn numpy
```

## 🎓 Educational Value

This tool demonstrates key NLP concepts:

- **Text Preprocessing**: Cleaning and tokenization
- **Feature Extraction**: TF-IDF vectorization
- **Similarity Measures**: Cosine similarity
- **Ranking Algorithms**: Sentence importance scoring
- **Evaluation Metrics**: Compression ratios and statistics

## 🔮 Future Enhancements

Potential improvements for advanced users:

- **Transformer Models**: Integration with BERT, GPT for abstractive summarization
- **Web Scraping**: Direct URL processing with newspaper3k
- **Multiple Languages**: Support for non-English text
- **GUI Interface**: Desktop application with tkinter
- **API Endpoint**: REST API for web integration
- **Export Options**: PDF, Word document output

## 📖 References

- TF-IDF: [Term Frequency-Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- Cosine Similarity: [Cosine Similarity Measure](https://en.wikipedia.org/wiki/Cosine_similarity)
- NLTK Documentation: [Natural Language Toolkit](https://www.nltk.org/)
- Scikit-learn: [Machine Learning Library](https://scikit-learn.org/)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- New summarization algorithms
- Performance improvements
- Documentation enhancements
- Additional features

## 💡 Tips for Best Results

1. **Input Quality**: Use well-structured text with clear sentences
2. **Method Selection**: Use TF-IDF for technical content, frequency for general text
3. **Summary Length**: Start with 30% ratio and adjust based on needs
4. **Text Length**: Works best with 500+ word documents
5. **Preprocessing**: Ensure text is clean and properly formatted

---

**Created for**: Internship Project - Text Summarization Tool  
**Author**: Harshitha.Y.J  
**Date**: 09-07-2025  
**Version**: 1.0
