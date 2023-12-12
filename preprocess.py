import pandas as pd
from convokit import Corpus, download, Utterance
import re
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split


#create dataset made up of different subreddits
def create_dataset(subreddits, corpora):
    texts = []
    num_comments = []
    scores = []
    gilded_status = []

    # Iterate over each subreddit
    for subreddit in subreddits:
        # Get the utterances from the specified subreddit
        subreddit_utterances = corpora[subreddit].utterances
        
        # Iterate over each utterance in the subreddit
        for utterance in subreddit_utterances.values():
            if not isinstance(utterance, Utterance):
                continue

            # Skip utterance if text is "[deleted]" or "[removed]" or empty
            if utterance.text in ["[deleted]", "[removed]"] or not utterance.text.strip():
                continue

            # Clean the text by replacing tabs with a single space and removing extra whitespace
            cleaned_text = re.sub(r'\t', ' ', utterance.text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            # Check if the cleaned text is empty
            if not cleaned_text:
                continue

            # Check if the text has at least 10 characters
            if len(cleaned_text) < 10:
                continue
            
            # Check if the utterance is original post
            if utterance.reply_to is None:
                texts.append(cleaned_text)
                num_comments.append(0) 
                scores.append(utterance.meta['score'])
                gilded_status.append(utterance.meta['gilded'])
            else:
                # Handle the case where reply_to is not None
                texts.append(cleaned_text)
                num_comments.append(len(utterance.reply_to))
                scores.append(utterance.meta['score'])
                gilded_status.append(utterance.meta['gilded'])

    # Create a DataFrame from the lists
    data = pd.DataFrame({
        'Text': texts,
        'Num_Comments': num_comments,
        'Score': scores,
        'Gilded_Status': gilded_status
    })

    # save the data
    data.to_csv('combined_dataset.csv', index=False)


# Function to get vader sentiment scores
def get_sentiment_scores(text):
    # Initialize vader
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']


#run vader sentiment analysis on data
def vader(data):
    train_data, sub_data = train_test_split(data, test_size=0.2, random_state=42)
    unlabel_data, test_data = train_test_split(sub_data, test_size=0.001, random_state=42)

    train_data['Sentiment'] = train_data['Text'].apply(get_sentiment_scores)
    import pdb; pdb.set_trace()

    # Save the labeled data to a CSV file
    train_data.to_csv('auto_labeled_dataset.csv', index=False)
    test_data.to_csv('hand_labeled_dataset.csv', index=False)
    unlabel_data.to_csv('unlabeled_dataset.csv', index=False)


def main():
    # Download individual subreddits
    subreddits = ['News_Politics', 'CanadaPolitics', 'AmItheAsshole']
    corpora = {}

    for subreddit in subreddits:
        corpora[subreddit] = Corpus(download(f"subreddit-{subreddit}"))

    create_dataset(subreddits, corpora)
    data = pd.read_csv('to_label_1.csv')

    vader(data)
    

if __name__ == "__main__":
    main()