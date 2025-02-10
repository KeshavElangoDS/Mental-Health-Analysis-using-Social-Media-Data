# Mental Health Analysis using Social Media Data

### Keshav Elango (RUID: 227002518)

## Github repository:

https://github.com/KeshavElangoDS/Mental-Health-Analysis-using-Social-Media-Data

## Problem:

This project seeks to apply machine learning techniques to identify and classify social media posts related to mental health issues. Social media platforms have become important spaces where individuals openly discuss personal challenges, making it important to detect these conversations for timely interventions. 

By developing a model that can distinguish between mental health-related posts and others, the project aims to categorize these discussions into specific mental health conditions, such as anxiety, depression, and other disorders. This methodology holds the potential to contribute to early identification and support, offering valuable insights into prevalent mental health topics and facilitating better-targeted interventions.

### Key Steps:

This project involves several key steps, starting with the collection of data from social media platforms such as Twitter and Reddit. 
The collected data is then cleaned and preprocessed to ensure its relevance and accuracy for analysis. 
Text understanding is enhanced through natural language processing (NLP) techniques, using tools like spaCy for part-of-speech tagging and BERT for capturing the contextual meaning of words. 
To convert text data into meaningful features, techniques such as CountVectorizer and TF-IDF are employed. 
Finally, machine learning models like XGBClassifier, LSTM, and RNNs are used to classify the posts based on their relevance to mental health topics, facilitating the identification and categorization of these discussions.

## Data Sources:

Data sources for this project include Kaggle datasets, the Twitter X API, the PushShift API for Reddit, and additional resources from MIT Libraries and OpenICPSR, which provide comprehensive sentiment analysis datasets focused on mental health-related discussions.

Data is gathered from Kaggle or corresponding API for Twitter and Reddit from the TwitterAPI/PushShiftAPI
The data mostly would consists of keywords such as ‘anxiety,’ ‘depression,’ and ‘stress. r/depression, r/Anxiety,
r/bipolar, r/BPD, r/schizophrenia, and r/autism.

Some of the datasets for consideration are 

Sentiment Analysis Mental Health Tweets 2017-2023
https://www.kaggle.com/datasets/zoegreenslade/twittermhcampaignsentmentanalysis?select=MH_Campaigns1723.csv

Sentiment Analysis for Mental Health
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health?select=Combined+Data.csv

OpenICPSR Data: 
https://www.openicpsr.org/openicpsr/project/175582/version/V1/view

MIT Libraries : 
https://rdi.libraries.mit.edu/record/zenodo:3941387 , https://zenodo.org/records/3941387

