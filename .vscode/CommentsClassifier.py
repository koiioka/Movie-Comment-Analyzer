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

# 使用示例数据格式要求
# 文件格式：

# 评论数据应以文本文件（.txt）的形式存储在指定的文件夹中。
# 每个评论应单独存储在一个文本文件中，以便程序逐个读取和处理。
# 文件命名：

# # 文本文件可以按任意命名规则命名，如comment1.txt, comment2.txt等，确保文件扩展名为.txt。
# # 内容格式：
# 
# # 每个文本文件应包含一条完整的评论。确保评论文本清晰且无多余的格式字符（如HTML标签、特殊字符等）。
# # 文本应为纯文本格式，避免使用复杂的格式（如富文本格式）。

data_folder = "path_to_comments_folder"
output_folder = "path_to_output_folder"
main(data_folder, output_folder)