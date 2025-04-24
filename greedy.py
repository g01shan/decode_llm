"""Implements greedy decoding of the a huggingface model."""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate


def parse_cli_arguments():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description="Greedy decoding a HF model!")
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="+",
        default=["Tehran is", "Sing to me one song for joy and"],
        help="User prompts.",
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="gpt2",
        help="The model id from huggingface hub.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum length of prompt + response.",
    )
    return parser.parse_args()


def get_the_best_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hf_greedy_response(model, input_ids, attention_mask, max_length):
    """Computes the greedy decoding response using HF library."""
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=1,
        do_sample=False,
    )


def get_greedy_response_without_kv_cache(model, input_ids, attention_mask, max_legnth):
    """Computes the greedy decoding response using our own implementation!"""
    seq_length = input_ids.shape[1]
    while seq_length < max_legnth:
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=1) - 1, min=0)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
        last_token_logits = logits[:, -1, :]
        # find the next best tokens
        next_token_ids = last_token_logits.argmax(dim=1).unsqueeze(1)
        input_ids = torch.concat((input_ids, next_token_ids), dim=1)
        attention_mask = torch.concat(
            (attention_mask, torch.ones_like(next_token_ids)), dim=1
        )
        seq_length = input_ids.shape[1]
    return input_ids


def get_greedy_response_with_kv_cache(model, input_ids, attention_mask, max_legnth):
    """Computes the greedy decoding response using our own implementation!"""
    seq_length = input_ids.shape[1]
    if seq_length >= max_legnth:
        return input_ids
    # prefill
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=1) - 1, min=0)
    model_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    kv_cache = model_output.past_key_values
    logits = model_output.logits
    # moving forward we only need position_ids for the last tokens
    position_ids = position_ids[:, -1:]
    while seq_length < max_legnth:
        last_token_logits = logits[:, -1, :]
        # find the next best tokens
        next_token_ids = last_token_logits.argmax(dim=1).unsqueeze(1)
        input_ids = torch.concat((input_ids, next_token_ids), dim=1)
        attention_mask = torch.concat(
            (attention_mask, torch.ones_like(next_token_ids)), dim=1
        )
        seq_length = input_ids.shape[1]
        # incrementing the position ids
        position_ids = position_ids + 1
        model_output = model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )
        logits = model_output.logits
        kv_cache = model_output.past_key_values
    return input_ids


def main():
    """Tests the greedy decoding implementation."""
    args = parse_cli_arguments()
    device = get_the_best_device()
    print(f"- We are running this on {device}!\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()  # The model must be in eval mode by default, but let's be safe!
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # tokenizing the prompt and preparing the model input
    tokenized_batch = tokenizer(
        args.prompts, return_tensors="pt", padding=True, padding_side="left"
    )
    with torch.no_grad():
        input_ids = tokenized_batch["input_ids"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)
        # running HuggingFace's greedy decoding
        start = time.perf_counter()
        hf_output = get_hf_greedy_response(
            model, input_ids, attention_mask, args.max_length
        )
        end = time.perf_counter()
        hf_decoded_output = tokenizer.batch_decode(hf_output, skip_special_tokens=True)
        print(
            "Greedy decoding results from HuggingFace --",
            f"Completed in {(end - start):.2f} seconds",
        )
        print(tabulate(zip(args.prompts, hf_decoded_output), tablefmt="grid"))
        print()
        # running our own greedy decoding (without kv cache)
        start = time.perf_counter()
        output = get_greedy_response_without_kv_cache(
            model, input_ids, attention_mask, args.max_length
        )
        end = time.perf_counter()
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(
            "Greedy decoding results from us (with KVout cache) --",
            f"Completed in {(end - start):.2f} seconds",
        )
        print(tabulate(zip(args.prompts, decoded_output), tablefmt="grid"))
        print()
        # running our own greedy decoding (with kv cache)
        start = time.perf_counter()
        output_with_kv = get_greedy_response_with_kv_cache(
            model, input_ids, attention_mask, args.max_length
        )
        end = time.perf_counter()
        decoded_output_with_kv = tokenizer.batch_decode(
            output_with_kv, skip_special_tokens=True
        )
        print(
            "Greedy decoding results from us (with KV cache) --",
            f"Completed in {(end - start):.2f} seconds",
        )
        print(tabulate(zip(args.prompts, decoded_output_with_kv), tablefmt="grid"))


if __name__ == "__main__":
    main()
