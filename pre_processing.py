import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from unidecode import unidecode
import contractions
from urlextract import URLExtract
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import datetime
import multiprocessing as mp

# https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset?select=Tweets.csv

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)

# Initialize URL extractor and stopwords
extractor = URLExtract()
stop_words = set(stopwords.words("english"))
english_words = set(words.words())


def preprocess_tweet(tweet: str) -> str:
    # Convert to ASCII
    tweet = unidecode(tweet)

    # Remove URLs
    urls = extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, "")

    # Remove numbers
    tweet = re.sub(r"\d+", "", tweet)

    # Expand contractions
    tweet = contractions.fix(tweet)

    # Transform @user to USER
    tweet = re.sub(r"@[^\s]+", "USER", tweet)

    # Tokenize
    tokens = word_tokenize(tweet.lower())

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word.isalnum()]

    return " ".join(token for token in tokens)


def process_chunk(chunk):
    return chunk.apply(preprocess_tweet)


if __name__ == "__main__":
    initial = datetime.datetime.now()

    # Load dataset
    size = 28000
    root = "DLE/assessment_1"
    df = pd.read_csv(f"{root}/raw_tweets.csv")
    df['selected_text'] = df['selected_text'].astype(str)
    df['sentiment'] = df['sentiment'].astype(str)
    # df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

    # Determine the number of processes to use
    num_processes = min(
        mp.cpu_count(), 15
    )  # Use up to 15 cores or all available if less

    # Split the dataframe into chunks
    chunks = np.array_split(df["selected_text"][:size], num_processes)

    # Process the data using multiprocessing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chunks"
        ):
            results.append(future.result())

    # Combine the results
    df["processed_text"] = pd.concat(results)

    print(f"Total processing time: {datetime.datetime.now() - initial}")

    df['target'] = df['sentiment']
    df['text'] = df['processed_text']

    # Save processed data
    df[["text", "processed_text", "target"]].to_csv(
        f"{root}/processed_tweets.csv", index=False
    )
