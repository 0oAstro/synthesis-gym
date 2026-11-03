"""Clean-only baseline: rejection-sampling fine-tuning (RFT) on the model's own successful episodes.

Takes base-model rollouts on SynthGym-v0/train (results/transcripts_<rollouts>.json + episodes CSV),
keeps only episodes that reached the target (w_target >= 0.9, no format errors), and fine-tunes LoRA on
the assistant turns. Failures are thrown away, which is what training on published (successful) recipes
amounts to. Same base model, same LoRA shape and a matched rollout budget to the GRPO run.
Usage (GPU box): accelerate launch --num_processes 4 scripts/train_rft.py --rollouts qwen3-4b-rollouts-train
"""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from transformers.utils import get_json_schema
from trl import SFTConfig, SFTTrainer

from synthgym.tool_env import SynthesisToolEnv

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_tokenizer(model: str):
    spec = importlib.util.spec_from_file_location("train_tools", ROOT / "scripts" / "train_grpo_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tokenizer(model)


def clean_messages(transcript: list[dict]) -> list[dict]:
    messages = []
    for message in transcript:
        entry = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            entry["tool_calls"] = [{"type": "function", "function": {
                "name": call["function"]["name"],
                "arguments": json.dumps(call["function"]["arguments"]),
            }} for call in message["tool_calls"]]
        if message["role"] == "tool":
            entry["name"] = message.get("name", "")
        messages.append(entry)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--rollouts", default="qwen3-4b-rollouts-train")
    parser.add_argument("--output", default="checkpoints/qwen-rft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    episodes = pd.read_csv(RESULTS / f"episodes_{args.rollouts}.csv")
    transcripts = json.loads((RESULTS / f"transcripts_{args.rollouts}.json").read_text())
    keep = episodes[(episodes["w_target"] >= 0.9) & (episodes["format_errors"] == 0)]["task_id"]
    env = SynthesisToolEnv()
    tools = [json.dumps(get_json_schema(method)) for method in [env.add_precursors, env.heat, env.regrind, env.xrd, env.stop]]
    rows = [{"messages": clean_messages(transcripts[task_id]), "tools": [json.loads(t) for t in tools]} for task_id in keep]
    print(f"clean-only dataset: {len(rows)} successful episodes out of {len(episodes)} rollouts", flush=True)
    dataset = Dataset.from_list([{"messages": json.dumps(r["messages"]), "tools": json.dumps(r["tools"])} for r in rows])
    dataset = dataset.map(lambda row: {"messages": json.loads(row["messages"]), "tools": json.loads(row["tools"])})

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_length=2560,
        assistant_only_loss=True,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        model_init_kwargs={"dtype": "float16", "attn_implementation": "sdpa"},
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset, processing_class=load_tokenizer(args.model), peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output + "/final")


if __name__ == "__main__":
    main()
