import pandas as pd
import matplotlib.pyplot as plt

class AccuracyCalculator:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_defined_topics = ["plot", "special effects", "acting", "costume design"]

    def calculate_accuracy(self):
        correct_predictions = 0
        total_predictions = 0

        for _, row in self.df.iterrows():
            original_sentiment = row['sentiment']
            for topic in self.user_defined_topics:
                detected_sentiment = row[topic]
                if detected_sentiment != "Unclassified":
                    total_predictions += 1
                    if detected_sentiment.lower() == original_sentiment.lower():  
                        correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy, total_predictions

    def generate_accuracy_plot(self, accuracy, total_predictions):
        plt.bar(['Accuracy'], [accuracy])
        plt.ylim(0, 1)  
        plt.xlabel('Metric')
        plt.ylabel('Rate')
        plt.title(f'Sentiment Analysis Accuracy (Total Predictions: {total_predictions})')
        plt.show()


if __name__ == "__main__":
    file_path = "defualt.csv" 
    accuracy_calculator = AccuracyCalculator(file_path)
    accuracy, total_predictions = accuracy_calculator.calculate_accuracy()
    print(f"Sentiment Analysis Accuracy: {accuracy:.2%}")
    print(f"Total Predictions: {total_predictions}")
    

    accuracy_calculator.generate_accuracy_plot(accuracy, total_predictions)