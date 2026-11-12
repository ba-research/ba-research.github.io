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
    'Alberta pipeline: incident review': ('On April 16, 2024, near Edson, Alberta, the Calgary control centre observed a drop in Emerson Creek compressor station discharge pressure at 10:48. At 10:50, a station operator reported an explosion and fire and suspected a rupture of the Grande Prairie Mainline. The Transportation Safety Board reports that the controller decided to shut down and isolate the suspected ruptured section.', 'Which operational team should review this incident report?', 'Pipeline control and emergency response\nPipeline integrity engineering\nInstrumentation maintenance', 'Source: [Transportation Safety Board of Canada · investigation P24H0018](https://www.tsb.gc.ca/eng/rapports-reports/pipeline/2024/p24h0018/p24h0018.html) · Retrospective research question; Qev was not used in the incident.'),
    'U.S. payment diversion: financial review': ('In April 2022, a U.S. company that regularly purchased equipment from a business partner received an email requesting a change in payment information. According to the U.S. Department of Justice, the sender was a fraudster posing as the partner, and the company wired $2,462,000 to the identified account.', 'Which review should receive this incident report?', 'Accounts payable and fraud investigation\nVendor account maintenance\nProcurement and supplier management', 'Source: [U.S. Department of Justice · October 31, 2023](https://www.justice.gov/usao-mdfl/pr/united-states-recovers-24-million-obtained-business-email-compromise) · Retrospective research question; Qev was not used in the incident.'),
    'Cloud authentication: support routing': ('On May 3, 2024, Google Cloud reported Firebase Authentication and Identity Platform authentication failures affecting some customers. Its engineering team identified a mitigation strategy and was implementing it. The service health update reported that the issue was resolved for affected users by 15:27 Pacific time.', 'Which support team should investigate reports matching this incident?', 'Identity and authentication incident support\nApplication support\nCloud network operations', 'Source: [Google Cloud Service Health · May 3, 2024](https://status.cloud.google.com/incidents/rErZ4r752o4rt4qaotfL) · Retrospective research question; Qev was not used in the incident.'),
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
    first = SCENARIOS["Alberta pipeline: incident review"]
    with gr.Blocks(title="Try Qev locally") as demo:
        gr.Markdown("# Try Qev locally\nExperimental decisions from BA Research. Inference uses your local files. Examples summarize public reports; questions are retrospective research prompts. Do not use results for safety-critical action without qualified review.")
        example = gr.Dropdown(list(SCENARIOS), value="Alberta pipeline: incident review", label="Load a scenario")
        state = gr.Textbox(value=first[0], label="Context", lines=4)
        question = gr.Textbox(value=first[1], label="Question")
        options = gr.Textbox(value=first[2], label="Choices, one per line", lines=4)
        source = gr.Markdown(value=first[3])
        example.change(lambda key: SCENARIOS[key], example, [state, question, options, source])
        button = gr.Button("Make a decision", variant="primary")
        result = gr.JSON(label="Decision")
        button.click(lambda s, q, o: decide(model, temperature, s, q, o), [state, question, options], result, concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
