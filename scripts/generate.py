#!/usr/bin/env python3
"""
Ad-hoc inference with a fine-tuned WX-AFD model.

Usage:
    # From a weather data file:
    python generate.py --model /path/to/merged/model --input weather.txt

    # From a validation example by index:
    python generate.py --model /path/to/merged/model --val_data val.jsonl --index 0

    # With custom generation parameters:
    python generate.py --model /path/to/merged/model --input weather.txt \
        --temperature 0.5 --top_p 0.95 --max_new_tokens 2048
"""
import argparse
import json
import sys

from wx_afd import SYSTEM_PROMPT, load_model, generate_afd


def main():
    parser = argparse.ArgumentParser(description="Generate an AFD from weather data")
    parser.add_argument("--model", required=True,
                        help="Path to merged model or HF model ID")
    parser.add_argument("--input", default=None,
                        help="Path to a text file containing weather data")
    parser.add_argument("--val_data", default=None,
                        help="Path to val.jsonl to use a validation example")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of validation example to use (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output", default=None,
                        help="Write generated AFD to this file instead of stdout")
    args = parser.parse_args()

    # Get weather input
    if args.input:
        with open(args.input) as f:
            weather_input = f.read().strip()
    elif args.val_data:
        with open(args.val_data) as f:
            lines = f.readlines()
        if args.index >= len(lines):
            print(f"Error: index {args.index} out of range "
                  f"(file has {len(lines)} examples)", file=sys.stderr)
            sys.exit(1)
        ex = json.loads(lines[args.index])
        weather_input = ex["messages"][1]["content"]
        reference = ex["messages"][2]["content"]
    else:
        print("Error: provide either --input or --val_data", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load_model(args.model)

    # Generate
    print("Generating AFD...", file=sys.stderr)
    afd_text = generate_afd(model, tokenizer, weather_input,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(afd_text + "\n")
        print(f"Generated AFD written to {args.output}", file=sys.stderr)
    else:
        print(afd_text)

    # Show reference if using val data
    if args.val_data and not args.output:
        print("\n" + "=" * 60, file=sys.stderr)
        print("REFERENCE AFD:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(reference, file=sys.stderr)


if __name__ == "__main__":
    main()
