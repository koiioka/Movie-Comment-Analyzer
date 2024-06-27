import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 加载预训练的spaCy模型
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# 用户输入的主题和匹配度门槛
user_defined_topics = ["plot", "special effects", "acting", "costume design"]
matching_threshold = 0.2  # 用户可以设置的匹配度门槛

# 读取CSV数据
def load_data(csv_file):
    df = pd.read_csv(csv_file)
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

# 匹配评论到主题
def match_comments_to_topics(lda, vectorizer, documents, user_defined_topics, matching_threshold):
    topic_assignments = []
    topic_word_distributions = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    
    for doc, original_doc in zip(vectorizer.transform(documents), documents):
        topic_scores = lda.transform(doc)
        matched_topics = []
        for idx, score in enumerate(topic_scores[0]):
            if score >= matching_threshold:
                matched_topics.append(user_defined_topics[idx])
        if matched_topics:
            topic_assignments.append('; '.join(matched_topics))
        else:
            topic_assignments.append('Unclassified')
    
    return topic_assignments

# 保存结果到CSV文件
def save_results_to_csv(df, topic_assignments, output_file):
    df['category'] = topic_assignments
    df.to_csv(output_file, index=False)

# 主函数
def main(input_csv, output_csv):
    df = load_data(input_csv)
    documents = df['review'].tolist()
    preprocessed_docs = preprocess_data(documents)
    lda, vectorizer = lda_topic_extraction(preprocessed_docs, len(user_defined_topics))
    topic_assignments = match_comments_to_topics(lda, vectorizer, documents, user_defined_topics, matching_threshold)
    save_results_to_csv(df, topic_assignments, output_csv)

# 使用示例
if __name__ == "__main__":
    input_csv = "IMDB Dataset.csv"  # 输入CSV文件路径
    output_csv = "IMDB Dataset with categories.csv"  # 输出CSV文件路径
    main(input_csv, output_csv)