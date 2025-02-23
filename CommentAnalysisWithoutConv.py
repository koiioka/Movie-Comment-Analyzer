import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocess text
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# User-defined topics and matching threshold
user_defined_topics = ["Mars", "plot", "soundtrack", "actor"]
matching_threshold = 0.1  # Matching threshold that users can set
convolution_window_size = 50  # Convolution window size
topic_presence_threshold = 1  # Blurriness threshold(1 is equal to close)

# Detect garbage comments
def is_garbage(text):
    return len(text.split()) < 5  # Comments with fewer than 5 words are considered garbage

# Read CSV data and handle encoding issues
def load_data(csv_file, sample_size=None):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='latin1')
    if sample_size:
        df = df.head(sample_size)
    return df

# Preprocess data
def preprocess_data(documents):
    return [preprocess(doc) for doc in documents]

# Perform LDA topic extraction
def lda_topic_extraction(preprocessed_docs, num_topics):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preprocessed_docs)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(X)
    return lda, vectorizer

# Sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity >= 0.11:
        return "Positive"
    elif analysis.sentiment.polarity <= -0.11:
        return "Negative"
    else:
        return "Unclassified"

# Match comments to topics and analyze sentiments
def match_and_analyze_comments(lda, vectorizer, documents, user_defined_topics, matching_threshold):
    topic_assignments = []
    topic_word_distributions = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    
    for doc, original_doc in zip(vectorizer.transform(documents), documents):
        if is_garbage(original_doc):
            topic_assignments.append({topic: "Unclassified" for topic in user_defined_topics})
            continue
        
        topic_scores = lda.transform(doc)
        topic_sentiments = {topic: "Unclassified" for topic in user_defined_topics}
        
        for idx, score in enumerate(topic_scores[0]):
            if score >= matching_threshold:
                topic = user_defined_topics[idx]
                sentiment = analyze_sentiment(original_doc)
                topic_sentiments[topic] = sentiment
        
        topic_assignments.append(topic_sentiments)
    
    return topic_assignments

# Apply convolution filter
def apply_convolution_filter(topic_assignments, user_defined_topics, window_size, threshold):
    total_comments = len(topic_assignments)
    num_full_windows = total_comments // window_size
    remaining_comments_start_idx = num_full_windows * window_size
    
    window_assignments = []

    for i in range(num_full_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = topic_assignments[start_idx:end_idx]
        
        topic_counts = {topic: 0 for topic in user_defined_topics}
        
        for assignment in window:
            for topic in user_defined_topics:
                if assignment[topic] != "Unclassified":
                    topic_counts[topic] += 1
        
        for assignment in window:
            for topic in user_defined_topics:
                if topic_counts[topic] < threshold:
                    assignment[topic] = "Unclassified"
        
        window_assignments.extend(window)
    
    # Handle remaining comments
    remaining_window = topic_assignments[remaining_comments_start_idx:]
    if remaining_window:
        topic_counts = {topic: 0 for topic in user_defined_topics}
        
        for assignment in remaining_window:
            for topic in user_defined_topics:
                if assignment[topic] != "Unclassified":
                    topic_counts[topic] += 1
        
        for assignment in remaining_window:
            for topic in user_defined_topics:
                if topic_counts[topic] < threshold:
                    assignment[topic] = "Unclassified"
        
        window_assignments.extend(remaining_window)
    
    return window_assignments

# Save results to CSV file
def save_results_to_csv(df, topic_assignments, output_file):
    columns = ['review', 'sentiment'] + user_defined_topics
    results = []

    for i, row in df.iterrows():
        comment = row['review']
        sentiment = row['sentiment']
        row_result = [comment, sentiment]
        
        topic_result = topic_assignments[i]
        row_result.extend([topic_result[topic] for topic in user_defined_topics])
        
        results.append(row_result)
    
    result_df = pd.DataFrame(results, columns=columns)
    result_df.to_csv(output_file, index=False)

# Generate sentiment score bar chart
def generate_sentiment_plot(df):
    sentiment_score = {topic: 0 for topic in user_defined_topics}
    
    for topic in user_defined_topics:
        topic_sentiments = df[topic]
        for sentiment in topic_sentiments:
            if sentiment == "Positive":
                sentiment_score[topic] += 1
            elif sentiment == "Negative":
                sentiment_score[topic] -= 1

    plt.bar(sentiment_score.keys(), sentiment_score.values())
    plt.xlabel('Topics')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Score by Topic')
    plt.show()

# Calculate sentiment analysis accuracy
def calculate_accuracy(df):
    correct_predictions = 0
    total_predictions = 0
    
    for _, row in df.iterrows():
        original_sentiment = row['sentiment']
        for topic in user_defined_topics:
            if row[topic] != "Unclassified":
                total_predictions += 1
                if row[topic] == original_sentiment:
                    correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, total_predictions

# Generate accuracy bar chart
def generate_accuracy_plot(accuracy, total_predictions):
    plt.bar(['Accuracy'], [accuracy])
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.xlabel('Metric')
    plt.ylabel('Rate')
    plt.title(f'Sentiment Analysis Accuracy (Total Predictions: {total_predictions})')
    plt.show()

# Main function
def main(input_csv, output_csv, sample_size=None):
    print("Loading data...")
    df = load_data(input_csv, sample_size)
    documents = df['review'].tolist()
    
    print("Preprocessing data...")
    preprocessed_docs = preprocess_data(documents)
    
    print("Extracting topics with LDA...")
    lda, vectorizer = lda_topic_extraction(preprocessed_docs, len(user_defined_topics))
    
    print("Matching comments to topics and analyzing sentiments...")
    topic_assignments = match_and_analyze_comments(lda, vectorizer, documents, user_defined_topics, matching_threshold)
    
    print("Applying convolution filter...")
    topic_assignments = apply_convolution_filter(topic_assignments, user_defined_topics, convolution_window_size, topic_presence_threshold)
    
    print("Saving results to CSV...")
    save_results_to_csv(df, topic_assignments, output_csv)
    
    print("Generating sentiment plot...")
    result_df = pd.read_csv(output_csv)
    generate_sentiment_plot(result_df)
    
    print("Calculating accuracy...")
    accuracy, total_predictions = calculate_accuracy(result_df)
    print(f"Sentiment Analysis Accuracy: {accuracy:.2%}")
    
    print("Generating accuracy plot...")
    generate_accuracy_plot(accuracy, total_predictions)
    
    print("Processing complete.")

# Example usage
if __name__ == "__main__":
    input_csv = "IMDB Dataset big.csv"  # Input CSV file path
    output_csv = "Noc.csv"  # Output CSV file path
    sample_size = 5000  # Number of samples to process, set to None to process all data
    main(input_csv, output_csv, sample_size)
