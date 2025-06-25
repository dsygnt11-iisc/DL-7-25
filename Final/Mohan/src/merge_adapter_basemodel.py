from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

working_dir = "/Users/mohanpanakam/Documents/Finetuning/finetuning"

# Path to your base model and adapter
base_model_id = "ProsusAI/finbert"
adapter_path = f"{working_dir}/best_model_fold_4"  # or your adapter directory

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=3)

# Load adapter into base model
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge adapter weights into base model
merged_model = peft_model.merge_and_unload()

# Save merged model (now a standard Hugging Face model)
merged_model.save_pretrained(f"{working_dir}/goldbert", safe_serialization=False)
AutoTokenizer.from_pretrained(base_model_id).save_pretrained(f"{working_dir}/goldbert")