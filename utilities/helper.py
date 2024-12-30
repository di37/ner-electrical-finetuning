import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import numpy as np
import evaluate

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os
from datetime import datetime


from huggingface_hub import (
    HfApi,
    create_repo,
    login,
    ModelCard,
    CardData
)

def repository_exists(repo_name: str, token: str) -> bool:
    """Check if a repository already exists on the Hub."""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_name, token=token)
        return True
    except RepositoryNotFoundError:
        return False

def create_model_card(
    model_name: str,
    task_type: str,
    base_model: str,
    dataset_name: str,
    language: str = "en",
    metrics: dict = None,
    **kwargs  # Accept any additional kwargs
) -> ModelCard:
    """Create a detailed model card for the repository."""
    card_data = CardData(
        language=language,
        license="apache-2.0",
        library_name="transformers",
        tags=[task_type, base_model.split('/')[-1]],
        datasets=[dataset_name],
        metrics=metrics
    )
    
    content = f"""
---
language: {language}
license: mit
base_model: {base_model}
tags:
- {task_type}
- {base_model.split('/')[-1]}
datasets:
- {dataset_name}
metrics:
{chr(10).join(f'- {k}: {v}' for k, v in (metrics or {}).items())}
---

# {model_name}

## Model description

This model is fine-tuned from [{base_model}](https://huggingface.co/{base_model}) for {task_type} tasks.

## Training Data

The model was trained on the {dataset_name} dataset. 

## Model Details
- **Base Model:** {base_model}
- **Task:** {task_type}
- **Language:** {language}
- **Dataset:** {dataset_name}

## Training procedure

### Training hyperparameters
[Please add your training hyperparameters here]

## Evaluation results

{"### Metrics\\n" + "\\n".join(f"- {k}: {v}" for k, v in (metrics or {}).items()) if metrics else ""}

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModel.from_pretrained("{model_name}")
```

## Limitations and bias

[Add any known limitations or biases of the model]

## Training Infrastructure

[Add details about training infrastructure used]

## Last update

{datetime.now().strftime('%Y-%m-%d')}
"""
    
    return ModelCard(content)

def push_model_to_hub(
    repo_name: str,
    token: str,
    task_type: str,
    base_model: str,
    dataset_name: str,
    model_dir: str,
    private: bool = False,
    model_card_kwargs: dict = None
):
    """Push model files directly to the Hub without cloning."""
    try:
        # Login to Hugging Face
        login(token=token)
        api = HfApi()
        
        # Create repository if it doesn't exist
        exists = repository_exists(repo_name, token)
        if not exists:
            repo_url = create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            print(f"Created new repository: {repo_url}")
        else:
            print(f"Using existing repository: {repo_name}")

        # Create and push model card
        if model_card_kwargs is None:
            model_card_kwargs = {}
            
        # Prepare model card kwargs
        card_kwargs = {
            'model_name': repo_name,
            'task_type': task_type,
            'base_model': base_model,
            'dataset_name': dataset_name,
            'language': model_card_kwargs.get('language', 'en'),
            'metrics': model_card_kwargs.get('metrics', None)
        }
        
        model_card = create_model_card(**card_kwargs)
        
        try:
            model_card.push_to_hub(repo_name, token=token)
            print("Model card pushed successfully")
        except Exception as e:
            print(f"Warning: Could not push model card: {str(e)}")

        # Push model files
        if model_dir and os.path.exists(model_dir):
            print(f"Pushing model files from {model_dir}")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_name,
                token=token,
                commit_message="Upload model files"
            )
            print("Model files pushed successfully")
            
        return f"https://huggingface.co/{repo_name}"
        
    except Exception as e:
        print(f"Error pushing model to hub: {str(e)}")
        raise
    

def clean_and_group_entities(ner_results, min_score=0.40):
    """
    Cleans and groups named entity recognition (NER) results based on a minimum score threshold.
    
    Args:
        ner_results (list of dict): A list of dictionaries containing NER results. Each dictionary should have the keys:
            - "word" (str): The recognized word or token.
            - "entity_group" (str): The entity group or label.
            - "start" (int): The start position of the entity in the text.
            - "end" (int): The end position of the entity in the text.
            - "score" (float): The confidence score of the entity recognition.
        min_score (float, optional): The minimum score threshold for considering an entity. Defaults to 0.40.
    
    Returns:
        list of dict: A list of grouped entities that meet the minimum score threshold. Each dictionary contains:
            - "entity_group" (str): The entity group or label.
            - "word" (str): The concatenated word or token.
            - "start" (int): The start position of the entity in the text.
            - "end" (int): The end position of the entity in the text.
            - "score" (float): The minimum confidence score of the grouped entity.
    """
    grouped_entities = []
    current_entity = None

    for result in ner_results:
        # Skip entities with score below threshold
        if result["score"] < min_score:
            if current_entity:
                # Add current entity if it meets threshold
                if current_entity["score"] >= min_score:
                    grouped_entities.append(current_entity)
                current_entity = None
            continue

        word = result["word"].replace("##", "")  # Remove subword token markers
        
        if current_entity and result["entity_group"] == current_entity["entity_group"] and result["start"] == current_entity["end"]:
            # Continue the current entity
            current_entity["word"] += word
            current_entity["end"] = result["end"]
            current_entity["score"] = min(current_entity["score"], result["score"])
            
            # If combined score drops below threshold, discard the entity
            if current_entity["score"] < min_score:
                current_entity = None
        else:
            # Finalize the current entity if it meets threshold
            if current_entity and current_entity["score"] >= min_score:
                grouped_entities.append(current_entity)
            
            # Start a new entity
            current_entity = {
                "entity_group": result["entity_group"],
                "word": word,
                "start": result["start"],
                "end": result["end"],
                "score": result["score"]
            }

    # Add the last entity if it meets threshold
    if current_entity and current_entity["score"] >= min_score:
        grouped_entities.append(current_entity)

    return grouped_entities