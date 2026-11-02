"""Generate plans for SynthGym-v0/test-200 with an open model (optionally + LoRA) and score them.

Writes results/plans_<name>.json and results/episodes_<name>.csv (same columns as run_eval.py).
Usage (GPU box): python scripts/eval_plans.py --name qwen3-4b-base --model Qwen/Qwen3-4B-Instruct-2507 [--lora checkpoints/grpo/final]
Greedy decoding, one sample per task, so before/after are paired on the same 200 targets.
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthgym.evaluate import run_suite
from synthgym.plan import PlanAgent, plan_prompt
from synthgym.tasks import list_tasks, task_spec

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--suite", default="SynthGym-v0/test-200")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda:0")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
    model.eval()

    task_ids = list_tasks(args.suite)
    plans: dict[str, str] = {}
    for start in range(0, len(task_ids), args.batch):
        batch = task_ids[start: start + args.batch]
        texts = [tokenizer.apply_chat_template(plan_prompt(task_spec(t)), tokenize=False, add_generation_prompt=True) for t in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=448, do_sample=False)
        for task_id, row in zip(batch, output):
            plans[task_id] = tokenizer.decode(row[encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{start + len(batch)}/{len(task_ids)}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"plans_{args.name}.json").write_text(json.dumps(plans, indent=1))
    frame = run_suite(lambda: PlanAgent(args.name, plans), task_ids)
    frame.to_csv(RESULTS / f"episodes_{args.name}.csv", index=False)
    print(frame[["precursor_match", "recipe_match", "success_sim", "w_target", "total_reward"]].mean().to_string())


if __name__ == "__main__":
    main()
