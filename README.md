# CNN Sentiment Classificator

## Description
This project is a sentiment classification tool that uses a Convolutional Neural Network (CNN) to analyze and classify the sentiment of tweets. The model predicts whether the sentiment of a tweet is positive, negative, or neutral.

## Project Structure
- `main.ipynb`: The main Jupyter Notebook containing the code for data preprocessing and model training.
- `pre_processing.py`: A Python script containing the `preprocess_tweet` function used for tweet preprocessing.
- `data/`: Directory where the dataset should be placed.

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install `pipenv` if you haven't already:
    ```sh
    pip install pipenv
    ```

3. Create a virtual environment and install dependencies:
    ```sh
    pipenv install
    ```

4. Activate the virtual environment:
    ```sh
    pipenv shell
    ```

## Usage
1. Ensure you have the dataset in the `./data` directory.
2. Run the Jupyter Notebook:
    ```sh
    jupyter notebook main.ipynb
    ```
3. Follow the steps in the notebook to preprocess the data and train the model.

### Example
The following example demonstrates how the model processes and predicts the sentiment of a list of tweets:

```python
tweets = [
    "I'm really unhappy with my purchase.", 
    "This place is amazing!",
    "The event was neither good nor bad.",
    "I'm so frustrated with the delays.",
    "Had a great time with friends!",
    "The meeting was quite boring.",
    "I'm very upset with the outcome.",
    "This is the best experience I've ever had!",
    "The presentation was just fine.",
    "I'm extremely dissatisfied with the service.",
    "Feeling so happy and content!",
    "The performance was just okay."
]

predictions = []
for t in tweets:
    cleaned_text = preprocess_tweet(t)
    transformed = transform_to_model_input([cleaned_text])
    model_predictions = cnn_model.predict([transformed])

    prediction = {}
    prediction["tweet"] = t

    label_mapping = {
        "neutral": "neutral 😐",
        "positive": "positive 👍",
        "negative": "negative 👎",
    }


    for z in zip([k for k, v in label_mapping.items()], list(model_predictions[0])):
        prediction[label_mapping[z[0]]] = z[1]

    predictions.append(prediction)



predictions_df = pd.DataFrame(predictions)
predictions_df
```