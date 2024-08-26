# Sentiment Analysis and Topic Modeling with Convolutional Filtering

## Overview
This project implements a pipeline for sentiment analysis and topic modeling of user reviews using Natural Language Processing (NLP) techniques. The pipeline includes preprocessing, topic extraction using Latent Dirichlet Allocation (LDA), sentiment analysis with TextBlob, and convolutional filtering to refine topic assignments and filter out noise.

_**Explanation**: This section provides a high-level summary of the project, explaining the main objectives and what the project is about._

## Features
- **Text Preprocessing**: Lemmatization, stop words removal, and punctuation filtering using spaCy.
- **Topic Extraction**: Identification of user-defined topics using LDA from `scikit-learn`.
- **Sentiment Analysis**: Classification of reviews into sentiment categories such as "Very Positive," "Positive," "Neutral," "Negative," and "Very Negative."
- **Convolutional Filtering**: A novel approach that applies a sliding window convolution filter to smooth topic assignments and eliminate noise.
- **Visualization**: Generation of bar plots to visualize sentiment distribution and sentiment analysis accuracy.

_**Explanation**: This section highlights the key functionalities of the project, giving users an understanding of what the project can do._

## Project Structure
- `main.py`: The main script that orchestrates the entire analysis process.
- `preprocess.py`: Contains functions for text preprocessing.
- `topic_modeling.py`: Implements LDA-based topic extraction.
- `sentiment_analysis.py`: Contains sentiment analysis logic using TextBlob.
- `convolution_filter.py`: Implements the improved convolutional filtering method.
- `visualization.py`: Generates plots for sentiment distribution and accuracy.
- `data/`: Folder containing input CSV files with user reviews.
- `output/`: Folder where the processed results and plots are saved.

_**Explanation**: This section provides an overview of the project's file organization, so users know what each file or directory is for._

## Getting Started

### Prerequisites
Ensure that you have Python 3.x installed along with the following Python packages:
- `pandas`
- `spacy`
- `scikit-learn`
- `textblob`
- `matplotlib`
