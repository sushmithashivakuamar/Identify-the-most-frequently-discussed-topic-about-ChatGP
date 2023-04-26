import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Twitter_Data.csv')

# Define a function to preprocess the tweet text
def preprocess(text):
    if type(text) == float: # Skip if text is not a string
        return ''
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\S+|#\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a string
    text = ' '.join(tokens)
    return text

# Detect the language of each tweet text and translate non-English tweets to English
for i, row in df.iterrows():
    tweet_text = row['Tweet']
    try:
        language = detect(tweet_text)
        if language != 'en':
            tweet_text = translate_text(tweet_text, dest='en').text
        df.at[i, 'tweet_text'] = tweet_text
    except:
        pass

# Preprocess the tweet text
df['processed_text'] = df['tweet_text'].apply(preprocess)

# Concatenate all processed tweet text into a single string
all_tweets = ' '.join(df['processed_text'].astype(str))

# Split the string into words and count their frequency
word_counts = Counter(all_tweets.split())

# Get the 10 most common words and their frequencies
top_words = word_counts.most_common(10)

# Print the results
print('The 10 most frequently discussed topics on Twitter are:')
for i, (word, count) in enumerate(top_words):
    print(f'{i+1}. {word}: {count} mentions')

# Perform NMF topic modeling
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(df['processed_text'])
nmf_model = NMF(n_components=5, random_state=1)
nmf_model.fit(tfidf)
topic_names = {0: 'Topic 1', 1: 'Topic 2', 2: 'Topic 3', 3: 'Topic 4', 4: 'Topic 5'}

# Get the topic distribution for each tweet
topic_distributions = nmf_model.transform(tfidf)

# Create a dataframe to show the dominant topic and score for each tweet
dominant_topics = [topic_names[topic_distribution.argmax()] for topic_distribution in topic_distributions]
dominant_scores = [topic_distribution.max() for topic_distribution in topic_distributions]
df_topics = pd.DataFrame({'Dominant Topic': dominant_topics, 'Dominant Score': dominant_scores})

# Concatenate the topic dataframe with the original dataframe
df = pd.concat([df, df_topics], axis=1)

# Print the results
print('the dominant topic and the score')
print(df[['processed_text', 'Dominant Topic', 'Dominant Score']])

# Get count and percentage of tweets for each dominant topic
count_by_topic = df_topics.groupby('Dominant Topic').size().reset_index(name='Count')
count_by_topic['Percentage'] = count_by_topic['Count'] / len(df_topics) * 100

# Print the results
print('Count and percentage of tweets for each dominant topic:')
print(count_by_topic)

# Sort the tweets by their dominant score for each topic
for topic in topic_names.values():
    df_topic_tweets = df[df['Dominant Topic'] == topic].sort_values(by='Dominant Score', ascending=False)

# Group the tweets based on dominant topic and concatenate them into a single string
grouped_topics = df.copy()
grouped_topics['tweet_text'].fillna('', inplace=True)
grouped_topics = grouped_topics.groupby('Dominant Topic')['tweet_text'].apply(lambda x: ' '.join(x)).reset_index()
# Group tweets by dominant topic and join them into a single string
grouped_topics = df.groupby('Dominant Topic')['tweet_text'].apply(lambda x: ' '.join(str(i) for i in x)).reset_index()


# Replace NaN values with 0 for the 'Dominant Score' column
df['Dominant Score'].fillna(0, inplace=True)

# Sort the tweets by their dominant score for each topic
for topic in topic_names.values():
    df_topic_tweets = df[df['Dominant Topic'] == topic].sort_values(by='Dominant Score', ascending=False)
    print(f'Top 5 tweets for {topic}:')
    for i, row in df_topic_tweets.head(5).iterrows():
        print(f'{i+1}. {row["tweet_text"]}')
    print('\n')

# Sort the entire dataframe by dominant score
df_sorted = df.sort_values(by='Dominant Score', ascending=False)

# Print the top 5 tweets
print('Overall top 5 tweets:')
for i, row in df_sorted.head(5).iterrows():
    print(f'{i+1}. {row["tweet_text"]}')

#---


# Step 1: Get top terms for each topic
top_terms = []
for topic_idx, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[:-11:-1] # get indices of top 10 words
    top_words = [list(vectorizer.vocabulary_.keys())[i] for i in top_words_idx]  # map indices to words
    top_terms.append(', '.join(top_words))

# Step 2: Get highly-relevant tweets for each topic
top_tweets = []
for topic_name in topic_names.values():
    topic_tweets = df[df['Dominant Topic'] == topic_name].sort_values(by='Dominant Score', ascending=False)
    top_tweets.append(topic_tweets.head(10)['tweet_text'].tolist())

# Step 3: Create topic dataframe
topic_df = pd.DataFrame({
    'Topic Name': topic_names.values(),
    'Top Terms': top_terms,
    'Highly-Relevant Tweets': top_tweets
})

# Print topic dataframe
print(topic_df)

"""# Initialize TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, use_idf=True)

# Fit vectorizer on text data
tfidf = vectorizer.fit_transform(df['tweet_text'].astype(str))

# Extract top words for each topic
summary = []
for i, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [vectorizer.vocabulary_[i] for i in top_words_idx]
    top_tweets = [df.iloc[j]['tweet_text'] for j in np.argsort(topic)[-5:]]
    summary.append({'Topic': topic_names[i], 'Top Words': ', '.join(top_words), 'Top Tweets': '\n'.join(top_tweets)})
summary_df = pd.DataFrame(summary)
print(summary_df)
"""

import matplotlib.pyplot as plt

# Set the style of the plot
plt.style.use('ggplot')

# Create a bar chart showing the count of tweets for each dominant topic
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(count_by_topic['Dominant Topic'], count_by_topic['Count'])
ax.set_xlabel('Dominant Topic')
ax.set_ylabel('Number of Tweets')
ax.set_title('Number of Tweets by Dominant Topic')
plt.show()
