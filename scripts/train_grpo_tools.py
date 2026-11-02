"""Multi-turn GRPO on SynthesisGym: the model calls gym tools, sees each observation (including XRD), and acts again.

Uses TRL `environment_factory`; the reward is the gym's episode return.
Usage (GPU box): accelerate launch --num_processes 4 scripts/train_grpo_tools.py --model HuggingFaceTB/SmolLM3-3B
"""

import argparse
import random
import warnings

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import add_response_schema

from synthgym.tasks import list_tasks
from synthgym.tool_env import SynthesisToolEnv, task_prompt

warnings.filterwarnings("ignore")


SMOLLM3_ASSISTANT = """{{ "<|im_start|>assistant\\n" + content.lstrip("\\n") + "<|im_end|>\\n" }}"""
SMOLLM3_ASSISTANT_NO_THINK = """{{ "<|im_start|>assistant\\n" + "<think>\\n\\n</think>\\n" + content.lstrip("\\n") + "<|im_end|>\\n" }}"""
TOOL_CALLS = """{%- for tc in message.tool_calls or [] -%}{{ "<tool_call>\\n" + ({"name": tc.function.name, "arguments": tc.function.arguments} | tojson) + "\\n</tool_call>" }}{%- endfor -%}"""


def patch_smollm3(tokenizer):
    """SmolLM3's template drops assistant tool_calls; re-render them in Hermes format and parse like Qwen3."""
    from trl.chat_template_utils import qwen3_template
    template = tokenizer.chat_template
    assert SMOLLM3_ASSISTANT in template and SMOLLM3_ASSISTANT_NO_THINK in template
    template = template.replace(SMOLLM3_ASSISTANT, SMOLLM3_ASSISTANT.replace(' + "<|im_end|>\\n" }}', " }}") + TOOL_CALLS + '{{ "<|im_end|>\\n" }}')
    template = template.replace(SMOLLM3_ASSISTANT_NO_THINK, SMOLLM3_ASSISTANT_NO_THINK.replace(' + "<|im_end|>\\n" }}', " }}") + TOOL_CALLS + '{{ "<|im_end|>\\n" }}')
    tokenizer.chat_template = template
    tokenizer.response_template = qwen3_template
    return tokenizer


def load_tokenizer(model: str, parser_from: str | None = None):
    """Tokenizer with a tool-calling chat template and response parser TRL understands."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    if "SmolLM3" in model:
        return patch_smollm3(tokenizer)
    return add_response_schema(tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--parser-from", default=None, help="borrow the tool-call parser of a TRL-supported model")
    parser.add_argument("--output", default="checkpoints/grpo-tools")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--n-tasks", type=int, default=931)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--chat-template-kwargs", default="")
    parser.add_argument("--steps-per-generation", type=int, default=4)
    parser.add_argument("--vllm-host", default=None, help="TRL vllm-serve host; generation runs there")
    parser.add_argument("--colocate", action="store_true", help="run vLLM inside the trainer process, sleeping during backward")
    parser.add_argument("--colocate-memory", type=float, default=0.45)
    args = parser.parse_args()

    task_ids = list_tasks("SynthGym-v0/train")
    random.Random(0).shuffle(task_ids)
    dataset = Dataset.from_list([{"prompt": task_prompt(t), "task_id": t} for t in task_ids[: args.n_tasks]])
    template_kwargs = dict(item.split("=") for item in args.chat_template_kwargs.split(",") if item)
    template_kwargs = {key: value.lower() == "true" if value.lower() in ("true", "false") else value for key, value in template_kwargs.items()}

    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_generations=8,
        max_completion_length=1536,
        max_tool_calling_iterations=10,
        temperature=1.0,
        beta=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=20,
        report_to="none",
        log_completions=True,
        num_completions_to_print=1,
        model_init_kwargs={"dtype": "float16", "attn_implementation": "sdpa"},
        steps_per_generation=args.steps_per_generation,
        use_vllm=args.vllm_host is not None or args.colocate,
        vllm_mode="colocate" if args.colocate else "server",
        vllm_gpu_memory_utilization=args.colocate_memory,
        vllm_enable_sleep_mode=args.colocate,
        vllm_server_host=args.vllm_host or "127.0.0.1",
        vllm_server_port=8000,
        chat_template_kwargs=template_kwargs or None,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=load_tokenizer(args.model, args.parser_from),
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        environment_factory=SynthesisToolEnv,
    )
    trainer.train()
    trainer.save_model(args.output + "/final")


if __name__ == "__main__":
    main()
