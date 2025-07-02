#!/usr/bin/env python
# Fine-tuning FinBERT for Financial Sentiment Classification in Google Colab with LoRA and Export Options

# Step 1: Install required packages (for Google Colab users only)
# Uncomment and run in Colab
# %pip install -q transformers datasets peft accelerate bitsandbytes wandb scikit-learn

# Step 2: Import required libraries
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer, EvalPrediction, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os # Import the os module
from sklearn.utils.class_weight import compute_class_weight
import re
from nlpaug.augmenter.word import SynonymAug
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict

# Step 3: Login to Weights & Biases
# wandb.login() # Commented out for testing purposes, uncomment if you want to log to wandb

# Step 4: Load financial domain-specific model and tokenizer
model_id = "ProsusAI/finbert"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load base model
model_config = AutoConfig.from_pretrained(
    "ProsusAI/finbert",
    num_labels=3,
    hidden_dropout_prob=0.2,    # Increase dropout
    attention_probs_dropout_prob=0.2,
    classifier_dropout=0.3
)

model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    config=model_config
)

# Add this before lora_config to debug available modules
for name, module in model.named_modules():
    print(name)

print(f"Loaded model {model_id} with {model.num_parameters()} parameters.")
# Get various model dimensions
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of attention heads: {model.config.num_attention_heads}")
print(f"Number of hidden layers: {model.config.num_hidden_layers}")
print(f"Intermediate size: {model.config.intermediate_size}")

# Step 5: Apply LoRA configuration
lora_config = LoraConfig(
    r=16,               # Increased rank for more capacity
    lora_alpha=32,      # Increased alpha (usually 2*r)
    target_modules=["query", "value", "key", "dense"],  # Added key attention
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)

# After model = get_peft_model(base_model, lora_config)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel()}")

model.print_trainable_parameters()

# Step 6: Load and preprocess dataset (expects JSON with 'text' and 'label' fields)

# Print current working directory to verify file location
print(f"Current working directory: {os.getcwd()}")

# Check if files exist
train_file = "Labeled_bullionvault_articles.csv"
val_file = "Labeled_bullionvault_articles.csv"

if not os.path.exists(train_file):
    print(f"Error: {train_file} not found in {os.getcwd()}")
elif not os.path.exists(val_file):
    print(f"Error: {val_file} not found in {os.getcwd()}")
else:
    print(f"Found {train_file} and {val_file}.")
    try:
        dataset = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
        print("Dataset loaded successfully.")

        def tokenize(example):
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize)

        # Step 7: Set training arguments
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # Calculate class weights
        labels = df['label'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = dict(zip(np.unique(labels), class_weights))

        training_args = TrainingArguments(
            output_dir="./finbert_seq_cls",
            # Increase if you have more memory
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            # Add warmup steps for learning rate
            warmup_steps=500,
            # Add weight decay for regularization
            weight_decay=0.1,         # Could increase for stronger regularization
            # Rest of the arguments remain the same
            num_train_epochs=5,          # Could increase for better convergence
            eval_strategy="steps",        # Updated from evaluation_strategy
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            learning_rate=5e-5,          # Could try higher
            logging_steps=20,
            fp16=False,                   # Disabled for MPS compatibility
            report_to="wandb",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            class_weights=class_weight_dict
        )

        # Step 8: Define metrics for evaluation
        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            acc = accuracy_score(p.label_ids, preds)
            f1 = f1_score(p.label_ids, preds, average="macro")
            return {"accuracy": acc, "f1": f1}

        # Step 9: Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics
        )

        # Step 10: Start training
        trainer.train()

        # Step 11: Save model and tokenizer
        model.save_pretrained("./finbert_seq_cls")
        tokenizer.save_pretrained("./finbert_seq_cls")

        # Test the model with example texts
        print("\nTesting the model:")
        for text in test_texts:
            result = predict_sentiment(text, model, tokenizer, device)
            print(f"\nText: {result['text'][:100]}...")  # Show first 100 chars
            print(f"Sentiment: {result['sentiment']}")
            print(f"Logits: {result['logits']}")
            print(f"Probabilities: {result['probabilities']}")

        # Run the evaluation
        print("\nRunning model evaluation...")
        evaluation_metrics = evaluate_model(trainer, tokenized_dataset)
        
    #return trainer, model, tokenizer, evaluation_metrics

    except Exception as e:
        print(f"An error occurred during training/evaluation: {e}")

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    """Compute metrics for evaluation"""
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

def train_model(dataset, model, tokenizer, device):
    """Main training function"""
    try:
        # Calculate class weights directly from dataset
        labels = dataset['train']['label']
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        training_args = TrainingArguments(
            output_dir="./finbert_seq_cls",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.1,
            num_train_epochs=5,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            learning_rate=5e-5,
            logging_steps=20,
            fp16=False,
            report_to="wandb",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics  # Now defined
        )

        trainer.train()
        return trainer, model, tokenizer

    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None

def initialize_model():
    """Initialize the model and tokenizer"""
    model_id = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_config = AutoConfig.from_pretrained(
        "ProsusAI/finbert",
        num_labels=3,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.3
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        config=model_config
    )
    
    # Print model info
    print(f"Loaded model {model_id}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    
    return model, tokenizer

def setup_training(model, tokenizer, train_file, val_file):
    """Setup training configuration"""
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Training or validation file not found in {os.getcwd()}")
    
    dataset = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
    print("Dataset loaded successfully.")
    
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512)
    )
    
    return dataset, tokenized_dataset

def load_and_process_data(train_file, val_file):
    """Load and process the dataset"""
    try:
        dataset = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def setup_data():
    """Setup and validate data"""
    train_file = "Labeled_bullionvault_articles.csv"
    val_file = "Labeled_bullionvault_articles.csv"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Training or validation file not found in {os.getcwd()}")
        return None, None
    
    print(f"Found {train_file} and {val_file}.")
    dataset = load_and_process_data(train_file, val_file)
    
    if dataset is None:
        return None, None
        
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512)
    )
    
    return dataset, tokenized_dataset

def main():
    """Main execution function"""
    try:
        # Initialize model and tokenizer
        model, tokenizer = initialize_model()
        
        # Load and process data
        dataset, tokenized_dataset = setup_data()
        if dataset is None:
            return None, None, None, None
            
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Calculate class weights from dataset
        labels = dataset['train']['label']
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # Train model
        trainer, model, tokenizer = train_model(tokenized_dataset, model, tokenizer, device)
        
        if trainer is not None:
            evaluation_metrics = evaluate_model(trainer, tokenized_dataset)
            if evaluation_metrics is not None:
                export_model(model, tokenizer, device)
            return trainer, model, tokenizer, evaluation_metrics
        
        return None, None, None, None
        
    except Exception as e:
        print(f"Error in main: {e}")
        return None, None, None, None

# Step 12: Modified TorchScript export
def export_model(model, tokenizer, device):
    """Export model to TorchScript"""
    try:
        if model is None:
            raise ValueError("Model is None, cannot export")
            
        # Create a wrapper class for inference
        class FinBERTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                outputs = self.model(input_ids)
                return outputs.logits

        # Prepare example input
        example_input = tokenizer(
            "Example text for export",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = example_input["input_ids"].to(device)

        # Wrap model and prepare for export
        model.eval()  # Set to evaluation mode
        wrapped_model = FinBERTWrapper(model).to(device)
        
        # Export with wrapper
        with torch.no_grad():
            torch_script_model = torch.jit.trace(
                wrapped_model,
                input_ids,
                strict=False
            )
            torch.jit.save(torch_script_model, "./finbert_seq_cls/model_torchscript.pt")
            print("Model successfully exported to TorchScript")
            
    except Exception as e:
        print(f"Error during TorchScript export: {e}")

# Optional - Export to ONNX (uncomment below if needed and ensure required packages are installed)
# from transformers.onnx import export
# from transformers.onnx.features import FeaturesManager
# model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="sequence-classification")
# onnx_config = model_onnx_config(model.config)
# export(tokenizer, model, onnx_config, opset=13, output="./finbert_seq_cls/model.onnx")

# Step 13: Model Testing
def predict_sentiment(text, model, tokenizer, device):
    # Move model to specified device
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move input tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Convert to numpy for easier handling
    logits = logits.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Get predicted class
    predicted_class = np.argmax(logits, axis=1)[0]
    
    # Map to sentiment labels
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {
        "text": text,
        "sentiment": sentiment_map[predicted_class],
        "logits": logits[0],
        "probabilities": probs[0]
    }

# Add these imports at the top
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(trainer, tokenized_dataset):
    """Evaluate model performance with detailed metrics"""
    try:
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["train"])
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
        
        # Get predictions for confusion matrix
        train_preds = trainer.predict(tokenized_dataset["train"])
        val_preds = trainer.predict(tokenized_dataset["validation"])
        
        # Calculate detailed metrics
        train_classification = classification_report(
            train_preds.label_ids,
            np.argmax(train_preds.predictions, axis=1),
            target_names=['negative', 'neutral', 'positive'],
            digits=4
        )
        
        val_classification = classification_report(
            val_preds.label_ids,
            np.argmax(val_preds.predictions, axis=1),
            target_names=['negative', 'neutral', 'positive'],
            digits=4
        )
        
        # Print detailed results
        print("\n=== Training Set Results ===")
        print(f"Loss: {train_metrics['eval_loss']:.4f}")
        print(f"Accuracy: {train_metrics['eval_accuracy']:.4f}")
        print(f"F1 Score: {train_metrics['eval_f1']:.4f}")
        print("\nDetailed Classification Report:")
        print(train_classification)
        
        print("\n=== Validation Set Results ===")
        print(f"Loss: {val_metrics['eval_loss']:.4f}")
        print(f"Accuracy: {val_metrics['eval_accuracy']:.4f}")
        print(f"F1 Score: {val_metrics['eval_f1']:.4f}")
        print("\nDetailed Classification Report:")
        print(val_classification)
        
        # Return all metrics
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_classification': train_classification,
            'val_classification': val_classification
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

# Add to imports at top
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict

def setup_kfold_data(n_splits=5):
    """Setup data for k-fold cross validation"""
    try:
        train_file = "Labeled_bullionvault_articles.csv"
        
        if not os.path.exists(train_file):
            print(f"Error: {train_file} not found in {os.getcwd()}")
            return None
        
        # Load full dataset
        full_dataset = load_dataset("csv", data_files={"train": train_file})["train"]
        print("Dataset loaded successfully.")
        
        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Create folds
        fold_datasets = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
            # Split into train and validation
            train_dataset = Dataset.from_dict(full_dataset[train_idx])
            val_dataset = Dataset.from_dict(full_dataset[val_idx])
            
            # Create DatasetDict
            fold_dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
            
            # Tokenize
            tokenized_dataset = fold_dataset.map(
                lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512)
            )
            
            fold_datasets.append(tokenized_dataset)
            
        return fold_datasets
        
    except Exception as e:
        print(f"Error setting up k-fold data: {e}")
        return None

def train_and_evaluate_kfold(model, tokenizer, device, n_splits=5):
    """Train and evaluate model using k-fold cross validation"""
    try:
        # Setup k-fold data
        fold_datasets = setup_kfold_data(n_splits)
        if fold_datasets is None:
            return None, None, None, None
            
        fold_metrics = []
        best_fold_score = 0
        best_fold_trainer = None
        best_fold_model = None
        
        # Train and evaluate for each fold
        for fold, dataset in enumerate(fold_datasets):
            print(f"\n=== Training Fold {fold + 1}/{n_splits} ===")
            
            # Calculate class weights for this fold
            labels = dataset['train']['label']
            class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            
            # Train model
            trainer, model, tokenizer = train_model(dataset, model, tokenizer, device)
            
            if trainer is not None:
                # Evaluate model
                metrics = evaluate_model(trainer, dataset)
                if metrics is not None:
                    fold_metrics.append(metrics)
                    
                    # Track best performing fold
                    fold_score = metrics['val_metrics']['eval_f1']
                    if fold_score > best_fold_score:
                        best_fold_score = fold_score
                        best_fold_trainer = trainer
                        best_fold_model = model.clone()
        
        # Calculate average metrics across folds
        avg_metrics = calculate_average_metrics(fold_metrics)
        print("\n=== Average Metrics Across Folds ===")
        print(f"Validation F1: {avg_metrics['val_f1']:.4f}")
        print(f"Validation Accuracy: {avg_metrics['val_accuracy']:.4f}")
        
        return best_fold_trainer, best_fold_model, tokenizer, avg_metrics
        
    except Exception as e:
        print(f"Error in k-fold training: {e}")
        return None, None, None, None

def calculate_average_metrics(fold_metrics):
    """Calculate average metrics across folds"""
    if not fold_metrics:
        return None
        
    avg_metrics = {
        'val_f1': np.mean([m['val_metrics']['eval_f1'] for m in fold_metrics]),
        'val_accuracy': np.mean([m['val_metrics']['eval_accuracy'] for m in fold_metrics]),
        'val_loss': np.mean([m['val_metrics']['eval_loss'] for m in fold_metrics])
    }
    return avg_metrics

# Update main function to use k-fold
def main():
    """Main execution function"""
    try:
        # Initialize model and tokenizer
        model, tokenizer = initialize_model()
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Train and evaluate using k-fold
        trainer, model, tokenizer, metrics = train_and_evaluate_kfold(model, tokenizer, device)
        
        if trainer is not None and metrics is not None:
            # Export best model
            export_model(model, tokenizer, device)
            return trainer, model, tokenizer, metrics
        
        return None, None, None, None
        
    except Exception as e:
        print(f"Error in main: {e}")
        return None, None, None, None

# Step 12: Modified TorchScript export
def export_model(model, tokenizer, device):
    """Export model to TorchScript"""
    try:
        if model is None:
            raise ValueError("Model is None, cannot export")
            
        # Create a wrapper class for inference
        class FinBERTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                outputs = self.model(input_ids)
                return outputs.logits

        # Prepare example input
        example_input = tokenizer(
            "Example text for export",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = example_input["input_ids"].to(device)

        # Wrap model and prepare for export
        model.eval()  # Set to evaluation mode
        wrapped_model = FinBERTWrapper(model).to(device)
        
        # Export with wrapper
        with torch.no_grad():
            torch_script_model = torch.jit.trace(
                wrapped_model,
                input_ids,
                strict=False
            )
            torch.jit.save(torch_script_model, "./finbert_seq_cls/model_torchscript.pt")
            print("Model successfully exported to TorchScript")
            
    except Exception as e:
        print(f"Error during TorchScript export: {e}")

# Optional - Export to ONNX (uncomment below if needed and ensure required packages are installed)
# from transformers.onnx import export
# from transformers.onnx.features import FeaturesManager
# model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="sequence-classification")
# onnx_config = model_onnx_config(model.config)
# export(tokenizer, model, onnx_config, opset=13, output="./finbert_seq_cls/model.onnx")

# Step 13: Model Testing
def predict_sentiment(text, model, tokenizer, device):
    # Move model to specified device
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move input tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Convert to numpy for easier handling
    logits = logits.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Get predicted class
    predicted_class = np.argmax(logits, axis=1)[0]
    
    # Map to sentiment labels
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {
        "text": text,
        "sentiment": sentiment_map[predicted_class],
        "logits": logits[0],
        "probabilities": probs[0]
    }

# Move this to the bottom of the file
if __name__ == "__main__":
    trainer, model, tokenizer, evaluation_metrics = main()