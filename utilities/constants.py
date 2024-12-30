import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

DATASET_ID = "disham993/ElectricalNER"
MODEL_ID = "answerdotai/ModernBERT-large"

LOGS = "logs"
OUTPUT_DATASET_PATH = os.path.join(
    "data", "tokenized_electrical_ner_modernbert"
)  # "data"
OUTPUT_DIR = "models"
MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_ID)
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, f"electrical-ner-{MODEL_ID.split('/')[-1]}")

EVAL_STRATEGY = "epoch"
LEARNING_RATE = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 5
WEIGHT_DECAY = 0.01

LOCAL_MODELS = {
    "google-bert/bert-base-uncased": "electrical-ner-bert-base-uncased",
    "distilbert/distilbert-base-uncased": "electrical-ner-distilbert-base-uncased",
    "google-bert/bert-large-uncased": "electrical-ner-bert-large-uncased",
    "answerdotai/ModernBERT-base": "electrical-ner-ModernBERT-base",
    "answerdotai/ModernBERT-large": "electrical-ner-ModernBERT-large",
}

ONLINE_MODELS = {
    "google-bert/bert-base-uncased": "disham993/electrical-ner-bert-base",
    "distilbert/distilbert-base-uncased": "disham993/electrical-ner-distilbert-base",
    "google-bert/bert-large-uncased": "disham993/electrical-ner-bert-large",
    "answerdotai/ModernBERT-base": "disham993/electrical-ner-ModernBERT-base",
    "answerdotai/ModernBERT-large": "disham993/electrical-ner-ModernBERT-large",
}
