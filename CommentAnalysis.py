import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import matplotlib.pyplot as plt

# 加载预训练的spaCy模型
nlp = spacy.load('en_core_web_sm')

# 预处理文本
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# 用户输入的主题和匹配度门槛
user_defined_topics = ["Mars", "plot", "soundtrack", "actor"]
matching_threshold = 0.1  # 用户可以设置的匹配度门槛
convolution_window_size = 500  # 卷积窗口大小
topic_presence_threshold = 50  # 模糊阈值

# 检测垃圾评论
def is_garbage(text):
    return len(text.split()) < 5  # 评论少于5个单词被认为是垃圾评论

# 读取CSV数据并处理编码问题
def load_data(csv_file, sample_size=None):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='latin1')
    if sample_size:
        df = df.head(sample_size)
    return df

# 预处理数据
def preprocess_data(documents):
    return [preprocess(doc) for doc in documents]

# 进行LDA主题提取
def lda_topic_extraction(preprocessed_docs, num_topics):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preprocessed_docs)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(X)
    return lda, vectorizer

# 情感分析
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity >= 0.11:
        return "Positive"
    elif analysis.sentiment.polarity <= -0.11:
        return "Negative"
    else:
        return "Unclassified"

# 匹配评论到主题并进行情感分析
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

# 进行卷积操作
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
    
    # 处理剩余的评论
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

# 保存结果到CSV文件
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

# 生成情感统计柱状图
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

# 计算情感分析准确率
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

# 生成准确率柱状图
def generate_accuracy_plot(accuracy, total_predictions):
    plt.bar(['Accuracy'], [accuracy])
    plt.ylim(0, 1)  # 设置y轴范围为0到1
    plt.xlabel('Metric')
    plt.ylabel('Rate')
    plt.title(f'Sentiment Analysis Accuracy (Total Predictions: {total_predictions})')
    plt.show()

# 主函数
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

# 使用示例
if __name__ == "__main__":
    input_csv = "IMDB Dataset big.csv"  # 输入CSV文件路径
    output_csv = "IMDB Dataset with categories and sentiments.csv"  # 输出CSV文件路径
    sample_size = 5000  # 处理的样本数量，设置为None处理所有数据
    main(input_csv, output_csv, sample_size)