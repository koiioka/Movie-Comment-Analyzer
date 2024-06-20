import os
import json
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

# 读取评论数据
def load_data(data_folder):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

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
    topic_assignments = {topic: [] for topic in user_defined_topics}
    topic_word_distributions = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    
    for doc, original_doc in zip(vectorizer.transform(documents), documents):
        topic_scores = lda.transform(doc)
        for idx, score in enumerate(topic_scores[0]):
            if score >= matching_threshold:
                matched_topic = user_defined_topics[idx]
                topic_assignments[matched_topic].append(original_doc)
    
    return topic_assignments

# 保存结果到文件夹
def save_results_to_folders(topic_assignments, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for topic, comments in topic_assignments.items():
        topic_folder = os.path.join(output_folder, topic.replace(" ", "_"))
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)
        
        for i, comment in enumerate(comments):
            with open(os.path.join(topic_folder, f"comment_{i}.txt"), 'w', encoding='utf-8') as file:
                file.write(comment)

# 主函数
def main(data_folder, output_folder):
    documents = load_data(data_folder)
    preprocessed_docs = preprocess_data(documents)
    lda, vectorizer = lda_topic_extraction(preprocessed_docs, len(user_defined_topics))
    topic_assignments = match_comments_to_topics(lda, vectorizer, documents, user_defined_topics, matching_threshold)
    save_results_to_folders(topic_assignments, output_folder)

# 使用示例
data_folder = "path_to_comments_folder"
output_folder = "path_to_output_folder"
main(data_folder, output_folder)