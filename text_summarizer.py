import nltk
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess the input text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence endings
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        return text.strip()
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        sentences = sent_tokenize(text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
    
    def calculate_word_frequency(self, text):
        """Calculate word frequency scores"""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        freq_dist = Counter(words)
        max_freq = max(freq_dist.values()) if freq_dist else 1
        
        # Normalize frequencies
        for word in freq_dist:
            freq_dist[word] = freq_dist[word] / max_freq
        
        return freq_dist
    
    def score_sentences(self, sentences, word_freq):
        """Score sentences based on word frequencies"""
        sentence_scores = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            if len(words) > 0:
                score = sum(word_freq.get(word, 0) for word in words) / len(words)
                sentence_scores[sentence] = score
        
        return sentence_scores
    
    def tfidf_summarize(self, text, num_sentences=3):
        """Summarize using TF-IDF approach"""
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence similarity scores
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Score sentences based on similarity to all other sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_scores[sentence] = np.mean(similarity_matrix[i])
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in [s[0] for s in top_sentences]:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
    
    def frequency_summarize(self, text, num_sentences=3):
        """Summarize using word frequency approach"""
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Calculate word frequencies
        word_freq = self.calculate_word_frequency(text)
        
        # Score sentences
        sentence_scores = self.score_sentences(sentences, word_freq)
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in [s[0] for s in top_sentences]:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
    
    def summarize(self, text, method='tfidf', summary_ratio=0.3):
        """Main summarization function"""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Calculate number of sentences for summary
        sentences = self.extract_sentences(text)
        num_sentences = max(1, int(len(sentences) * summary_ratio))
        
        if method == 'tfidf':
            return self.tfidf_summarize(text, num_sentences)
        elif method == 'frequency':
            return self.frequency_summarize(text, num_sentences)
        else:
            raise ValueError("Method must be 'tfidf' or 'frequency'")
    
    def get_text_statistics(self, original_text, summary):
        """Get statistics about the summarization"""
        original_sentences = len(self.extract_sentences(original_text))
        summary_sentences = len(self.extract_sentences(summary))
        
        original_words = len(word_tokenize(original_text))
        summary_words = len(word_tokenize(summary))
        
        compression_ratio = (1 - summary_words / original_words) * 100
        
        return {
            'original_sentences': original_sentences,
            'summary_sentences': summary_sentences,
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': f"{compression_ratio:.1f}%"
        }

def main():
    # Initialize the summarizer
    summarizer = TextSummarizer()
    
    print("=" * 60)
    print("TEXT SUMMARIZATION TOOL")
    print("=" * 60)
    
    # Example usage with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents, any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term artificial intelligence is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.
    
    The scope of AI is disputed, with some researchers describing it as a narrow field focused on specific tasks, while others view it as a broad discipline encompassing machine learning, natural language processing, computer vision, and robotics. Modern AI techniques are pervasive and are too numerous to list here. Frequently, when a technique reaches mainstream use, it is no longer considered AI; this phenomenon is described as the AI effect.
    
    High-profile examples of AI include autonomous vehicles, such as drones and self-driving cars, medical diagnosis, creating art, proving mathematical theorems, playing games, search engines, online assistants, image recognition, spam filtering, predicting judicial decisions and targeting online advertisements. With social media sites overtaken by AI-generated content, many users have started to prefer human-generated content instead.
    
    Artificial intelligence was founded as an academic discipline in 1956, and in the years since it has experienced several waves of optimism, followed by disappointment and the loss of funding, followed by new approaches, success, and renewed funding. AI research has tried many different approaches, experienced many failures, and has slowly been making progress. For some problems, AI has been shown to be as good as or better than humans at completing tasks.
    """
    
    print("\nORIGINAL TEXT:")
    print("-" * 40)
    print(sample_text.strip())
    
    # Generate summaries using different methods
    print("\n\nSUMMARIZATION RESULTS:")
    print("=" * 60)
    
    # TF-IDF based summary
    print("\n1. TF-IDF METHOD:")
    print("-" * 20)
    tfidf_summary = summarizer.summarize(sample_text, method='tfidf', summary_ratio=0.3)
    print(tfidf_summary)
    
    # Frequency based summary
    print("\n2. FREQUENCY METHOD:")
    print("-" * 20)
    freq_summary = summarizer.summarize(sample_text, method='frequency', summary_ratio=0.3)
    print(freq_summary)
    
    # Show statistics
    print("\n\nSUMMARIZATION STATISTICS:")
    print("=" * 60)
    
    stats_tfidf = summarizer.get_text_statistics(sample_text, tfidf_summary)
    stats_freq = summarizer.get_text_statistics(sample_text, freq_summary)
    
    print(f"\nTF-IDF Method Stats:")
    print(f"  Original: {stats_tfidf['original_sentences']} sentences, {stats_tfidf['original_words']} words")
    print(f"  Summary: {stats_tfidf['summary_sentences']} sentences, {stats_tfidf['summary_words']} words")
    print(f"  Compression: {stats_tfidf['compression_ratio']}")
    
    print(f"\nFrequency Method Stats:")
    print(f"  Original: {stats_freq['original_sentences']} sentences, {stats_freq['original_words']} words")
    print(f"  Summary: {stats_freq['summary_sentences']} sentences, {stats_freq['summary_words']} words")
    print(f"  Compression: {stats_freq['compression_ratio']}")
    
    # Interactive mode
    print("\n\nINTERACTIVE MODE:")
    print("=" * 60)
    print("You can now input your own text to summarize!")
    print("Type 'quit' to exit the program.")
    
    while True:
        print("\nEnter your text (or 'quit' to exit):")
        user_input = input().strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Text Summarization Tool!")
            break
        
        if len(user_input) < 100:
            print("Please enter a longer text (at least 100 characters) for meaningful summarization.")
            continue
        
        print("\nChoose summarization method:")
        print("1. TF-IDF (recommended)")
        print("2. Frequency-based")
        
        method_choice = input("Enter choice (1 or 2): ").strip()
        method = 'tfidf' if method_choice == '1' else 'frequency'
        
        try:
            summary = summarizer.summarize(user_input, method=method)
            stats = summarizer.get_text_statistics(user_input, summary)
            
            print(f"\nSUMMARY ({method.upper()} method):")
            print("-" * 40)
            print(summary)
            
            print(f"\nSTATISTICS:")
            print(f"Compression: {stats['compression_ratio']} ({stats['original_words']} → {stats['summary_words']} words)")
            
        except Exception as e:
            print(f"Error processing text: {e}")

if __name__ == "__main__":
    main()
