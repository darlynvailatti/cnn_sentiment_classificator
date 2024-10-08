import re
import pandas as pd
import numpy as np
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from unidecode import unidecode
import contractions
from urlextract import URLExtract
from tqdm import tqdm

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)

# Initialize URL extractor and stopwords
extractor = URLExtract()
stop_words = set(stopwords.words("english"))
english_words = set(words.words("en"))

# Define the number of rows to read
size = 9000000  # Replace with the desired number of rows
data_dir = "data"  # Define your data directory


def extract_dataset_one():
    full_ = pd.read_csv(f"{data_dir}/dataset_1.csv", nrows=size, on_bad_lines='skip')
    df = pd.DataFrame()
    df["text"] = full_["selected_text"]
    df["sentiment"] = full_["sentiment"]
    return df


def extract_dataset_two():
    full_ = pd.read_csv(f"{data_dir}/dataset_2.csv", nrows=size, on_bad_lines='skip')
    df = pd.DataFrame()
    df["text"] = full_["SentimentText"].astype(str)
    df["sentiment"] = full_["Sentiment"]  # 0 = negative, 1 = positive
    df["sentiment"] = df["sentiment"].apply(lambda x: "negative" if x == 0 else "positive")
    return df


def extract_dataset_three():
    full_ = pd.read_csv(f"{data_dir}/dataset_3.csv", nrows=size, on_bad_lines='skip')
    df = pd.DataFrame()
    df["text"] = full_["text"].astype(str)
    df["sentiment"] = full_["airline_sentiment"]
    return df


def preprocess_tweet(input_tweet: str) -> str:

    if not input_tweet or not isinstance(input_tweet, str):
        return None

    try:
        # Convert to ASCII
        tweet = unidecode(input_tweet)

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

        # Remove stopwords, punctuation, and non-English words in a single step
        tokens = [
            word
            for word in tokens
            if word.isalnum() and word in english_words
        ]

        return " ".join(token for token in tokens)
    except Exception as e:
        print(f"Error processing tweet: {input_tweet}")
        print(e)
        return ""


def process_chunk(chunk):
    # Dummy processing function, replace with actual processing logic

    return chunk.apply(preprocess_tweet)


if __name__ == "__main__":
    initial = datetime.datetime.now()

    # Load dataset
    df = pd.DataFrame(columns=["text", "sentiment"])
    dataset_extractors = (
        extract_dataset_one,
        extract_dataset_two,
        extract_dataset_three,
    )
    for extractor in dataset_extractors:
        try:
            extracted_df = extractor()
            df = pd.concat([df, extracted_df], ignore_index=True)
        except FileNotFoundError:
            print("File not found for dataset: ", extractor.__name__)
            break
        except Exception as e:
            print("Error loading dataset: ", extractor.__name__)
            print(e)
            break

    # Determine the number of processes to use
    num_processes = min(
        mp.cpu_count(), 15
    )  # Use up to 15 cores or all available if less

    # Split the dataframe into chunks
    chunks = np.array_split(df["text"], num_processes)

    # Process the data using multiprocessing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chunks"
        ):
            results.append(future.result().reset_index(drop=True))

    # Combine the results
    df["processed_text"] = pd.concat(results).reset_index(drop=True)

    print(f"Total processing time: {datetime.datetime.now() - initial}")

    df["target"] = df["sentiment"]
    df["text"] = df["processed_text"]

    # Save processed data
    df[["text", "processed_text", "target"]].to_csv(
        f"{data_dir}/processed_tweets.csv", index=False
    )
