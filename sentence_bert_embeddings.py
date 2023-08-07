#!/usr/bin/env python
# coding: utf-8

import sys

from datasets import load_dataset
from datasets import Dataset

from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


import torch
import torch.nn.functional as F

import umap
import plotly.express as px
import pandas as pd



def load_data() -> (Dataset, Dataset, Dataset):
    """
    Loads the STS Benchmark dataset from the GLUE benchmark collection. 
    Extracts the data per each split.

    Returns:
        train_data (Dataset): The training split of the STS Benchmark dataset.
        validation_data (Dataset): The validation split of the STS Benchmark dataset.
        test_data (Dataset): The test split of the STS Benchmark dataset.
    """

    # Load the STS Benchmark dataset from glue
    sts_dataset = load_dataset("glue", "stsb")

    # Access the different splits of the dataset
    train_data = sts_dataset['train']
    validation_data = sts_dataset['validation']
    test_data = sts_dataset['test']
    
    return train_data, validation_data, test_data


def process_data(data: Dataset) -> (list[str], list[int]):
    """
    Processes the data to extract the sentences, the similarity scores and the sentence_ids.
    All sentences for each sample are added to a single list. All sentence2 sentences are appended after all the sentences1 sentences.

    Original data: 
    sample1_sentence1, sample1_sentence2
    ....
    sampleN_sentence1, sampleN_sentence2

    all_sentences:
    sample1_sentence1, ..., sampleN_sentence1, sample1_sentence2, ..., sampleN_sentence2

    Args:
        data (Dataset): List of dict input data.

    Returns:
        all_sentences (list[str]): List containing all the sentences from all data samples.
        ids (list[int]): The id for each sample. sentence1 and sentence2 that come frome the same sample have the same id
    """
    
    # Extracts all "sentences1" in a list
    sentences1 = [item['sentence1'] for item in data]
    
    # Extracts all "sentences2" in a list
    sentences2 = [item['sentence2'] for item in data]
    
    # Extract the similarity scores
    similarity_scores = [item['label'] for item in data]
    
    # Create a list containing all "sentences1" followed by all "sentences2" sample
    all_sentences = sentences1 + sentences2

    # Generate a sample id for each sentence. Sentences in the same sample share the same id.
    ids = list(range(0, len(data), 1)) + list(range(0, len(data), 1))
    
    return all_sentences, ids


def load_hf_model(model_name: str) -> (AutoTokenizer, AutoModel):
    """
    Loads the necessary HuggingFace tokenizer and embedding models. 

    Args: 
        model_name (str): The model name identifier in the HF model hub

    Returns:
        tokenizer (AutoTokenizer): The tokenizer model corresponding to the model_name.
        model (AutoModel): The transformers model (e.g., BERT, etc) corresponding to the model_name
    """

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name) 
        
        return tokenizer, model
    
    except (OSError, ValueError) as e:
        # Handle the exception if the model name is incorrect or not found
        print(f"An error occurred while loading the model: {str(e)}")
        sys.exit(1)


def mean_pooling(model_output: BaseModelOutputWithPoolingAndCrossAttentions, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs mean pooling on the token embeddings using an attention mask.

    Args:
        model_output (BaseModelOutputWithPoolingAndCrossAttentions]): Model output containing the token embeddings as the first element.
        attention_mask (torch.Tensor): Tensor containing the attention mask.

    Returns:
        torch.Tensor: Tensor containing the mean-pooled token embeddings.
    """
    
    
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def extract_sentence_embeddings(sentences: list[str], tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    try: 
        # Check if sentences is a non-empty list
        if not sentences or not isinstance(sentences, list):
            raise ValueError("Sentences must be a non-empty list.")
        
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
        return sentence_embeddings
     
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


def visualize_umap_embeddings(embeddings: torch.Tensor, sentences, labels, figure_name):
    """
    UMAP visualization of the sentence embeddings. 
    The function generates a 3D plot of the sentence embeddings. The actual sentence is visible when you hover over a point. 
    The sentenne ids are used as labels in the plot.

    Args:
        embeddings (torch.Tensor): Model output containing the token embeddings as the first element.
        sentences (list[str]): The sentences that correspond to each embedding representation.
        labels (list[int]): The sentence ids that correspong to the embedding representation.
        figure_name (str): The name of the figure to be saved.
    """
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    # DataFrame to hold the 3D coordinates and sentences
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df['sentence'] = sentences
    df['label'] = labels  # Optional: include labels if you have them

    # 3D scatter plot with hover information for sentences
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', hover_data={'sentence': True, 'x': False, 'y': False, 'z': False}) 

    # Save figure
    fig.write_html(figure_name + '.html')
    

def main():
    """
    Code that loads the STSB data, extracts sentence embeddings using the HuggingFace api and generates a 3D UMAP visualization of the embeddings. 
    """

    # Load the data
    train_data, validation_data, test_data = load_data()
    print("Train, validation and test data are loaded.")
    print(f"Number of {len(train_data)} rows in train data.")
    print(f"Number of {len(validation_data)} rows in validation data.")
    print(f"Number of {len(test_data)} rows in test data.")
    
    # Process the test data to extract all the sentences with their indices
    test_sentences, test_ids = process_data(test_data)
    
    # Load the tokenizer and the sentene embedding models
    sent_tokenizer, sent_model = load_hf_model('sentence-transformers/all-MiniLM-L6-v2') 
    print(f"\nAll 'sentence-transformers/all-MiniLM-L6-v2' models are loaded.")

    # Extract the sentene embeddings
    sent_embeddings = extract_sentence_embeddings(test_sentences, sent_tokenizer, sent_model)
    print("Sentence embeddings were extracted.")
    print("Embeddings dimension:", sent_embeddings.shape)

    # Visualize the embeddings using UMAP 3D transform
    visualize_umap_embeddings(sent_embeddings, test_sentences, test_ids, 'stsb_fig')
    print(f"UMAP visualization of embeddings stored in stsb_fig.html")
    print("DONE")


if __name__ == "__main__":
    main()