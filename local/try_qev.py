"""Run an existing, approved Qev checkpoint locally after dependencies are installed."""
import argparse
import json
import math
import os
from pathlib import Path

# Must be set before importing transformers or huggingface_hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch
from jev.api import compile_request, format_response
from jev.model import DecisionModel

SCENARIOS = {
    "Energy: maintenance triage": (
        "An Alberta operator reports an intermittent pressure sensor fault at a remote facility. The latest reading is inconsistent and a technician has not inspected it. No leak is confirmed.",
        "Which queue should review this report?",
        "Routine maintenance planning\nOperations team for prompt assessment\nClose without review",
    ),
    "Customer support: duplicate charge": (
        "A US customer reports two charges for one subscription renewal. Both transaction references are available and no refund has been confirmed.",
        "Which team should review this request?",
        "Billing and payments\nTechnical support\nSales",
    ),
    "AI routing: production incident": (
        "A database migration caused payment failures across a production service. An initial fix did not work and multiple interfaces are involved.",
        "Which task tier should be considered?",
        "Routine scoped task\nStandard implementation and debugging\nComplex cross-system investigation",
    ),
}


def checkpoint_info(path):
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Checkpoint must be a local directory.")
    config = json.loads((root / "model.json").read_text(encoding="utf-8"))
    base = Path(config.get("model_id", "")).expanduser()
    if not base.is_absolute() or not base.is_dir():
        raise ValueError("model.json must name an existing absolute local Qwen base directory as model_id. Hub IDs and remote downloads are disabled.")
    temperature = json.loads((root / "temperature.json").read_text(encoding="utf-8"))
    value = temperature.get("temperature")
    if temperature.get("split") != "calibration" or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid calibration temperature metadata.")
    return root, float(value)


def decide(model, temperature, state, question, options):
    choices = [line.strip() for line in options.splitlines() if line.strip()]
    if not state.strip() or not question.strip() or not 2 <= len(choices) <= 8 or len(set(choices)) != len(choices):
        raise ValueError("Provide context, a question, and 2–8 distinct choices.")
    records = compile_request(state, {"decision": {"type": "choice", "instructions": question, "criteria": {choice: None for choice in choices}}})
    with torch.inference_mode():
        logits = model(records)
        probabilities = [torch.softmax(row.double() / temperature, dim=-1).cpu().tolist() for row in logits]
    return format_response(records, probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory containing approved Qev adapter, head, model.json and temperature.json")
    parser.add_argument("--device", default="cpu", help="Local PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    root, temperature = checkpoint_info(args.checkpoint)
    print("Loading local checkpoint; the first run can take several minutes.", flush=True)
    model = DecisionModel.load(str(root), device=args.device)
    first = SCENARIOS["Energy: maintenance triage"]
    with gr.Blocks(title="Try Qev locally") as demo:
        gr.Markdown("# Try Qev locally\nExperimental decisions from BA Research. Inference uses your local files. Examples are fictional; do not use results for safety-critical action without qualified review.")
        example = gr.Dropdown(list(SCENARIOS), value="Energy: maintenance triage", label="Load a scenario")
        state = gr.Textbox(value=first[0], label="Context", lines=4)
        question = gr.Textbox(value=first[1], label="Question")
        options = gr.Textbox(value=first[2], label="Choices, one per line", lines=4)
        example.change(lambda key: SCENARIOS[key], example, [state, question, options])
        button = gr.Button("Make a decision", variant="primary")
        result = gr.JSON(label="Decision")
        button.click(lambda s, q, o: decide(model, temperature, s, q, o), [state, question, options], result, concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
