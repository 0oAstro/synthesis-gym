"""GRPO fine-tuning of an open model on SynthGym-v0/train in plan mode (TRL + LoRA).

Reward = the gym's episode return for the generated plan (synthgym.plan.score_plan).
Usage (GPU box): accelerate launch --num_processes 4 scripts/train_grpo.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import random
import warnings

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from synthgym.plan import plan_prompt, score_plan
from synthgym.tasks import list_tasks, task_spec

warnings.filterwarnings("ignore")


def gym_reward(prompts, completions, task_id, **kwargs) -> list[float]:
    rewards = []
    for completion, one_task in zip(completions, task_id):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(float(score_plan(task_spec(one_task), text)["reward"]))
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", default="checkpoints/grpo")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--n-tasks", type=int, default=600)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--accum", type=int, default=8)
    args = parser.parse_args()

    task_ids = list_tasks("SynthGym-v0/train")
    random.Random(0).shuffle(task_ids)
    task_ids = task_ids[: args.n_tasks]
    dataset = Dataset.from_list([{"prompt": plan_prompt(task_spec(t)), "task_id": t} for t in task_ids])

    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_generations=8,
        max_completion_length=384,
        temperature=1.0,
        beta=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=20,
        report_to="none",
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": "float16"},
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.model, reward_funcs=[gym_reward], args=config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output + "/final")


if __name__ == "__main__":
    main()
