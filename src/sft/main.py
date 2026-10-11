import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from .items import Item
from openai import OpenAI
import json
load_dotenv()

hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)

username = "ed-donner"
dataset_name = f"{username}/items_lite"
train, val, test = Item.from_hub(dataset_name)


client =OpenAI()

fine_tune_train = train[:100]
fine_tune_validation = val[:50]

def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"}
    ]

def make_jsonl(items):
    result=""
    for item in items:
        message=messages_for(item)
        message_str=json.dumps(message)
        result+= '{"messages":'+ message_str + '}\n'
    return result.strip()

def write_jsonl(items,filename):
    with open(filename,"w") as f:
        jsonl=make_jsonl(items)
        f.write(jsonl)

jsonl_dir = Path(__file__).resolve().parent / "jsonl"
jsonl_dir.mkdir(parents=True, exist_ok=True)
train_path = jsonl_dir / "fine_tune_train.jsonl"
validation_path = jsonl_dir / "fine_tune_validation.jsonl"

write_jsonl(fine_tune_train, train_path)
write_jsonl(fine_tune_validation, validation_path)

with train_path.open("rb") as f:
    train_file=client.files.create(file=f,purpose="fine-tune")

with validation_path.open("rb") as f:
    validation_file=client.files.create(file=f,purpose="fine-tune")



client.fine_tuning.jobs.create(
    model='gpt-4.1-nano-2025-04-14',
    training_file=train_file.id,
    validation_file=validation_file.id,
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},
    suffix="pricer"
)