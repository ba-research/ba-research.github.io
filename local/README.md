# Run Qev on your own machine

This is an experimental local decision playground. After installation, it loads an existing Qev checkpoint and Qwen base from local directories with Hugging Face offline mode enabled. It listens only on `127.0.0.1`; it does not create a public sharing link. Check the model card, applicable terms and your organization's approval before obtaining or using model files.

You need Python, enough RAM for your selected model, the Qev adapter/head/calibration files, and its matching Qwen base. Package installation and obtaining approved files require network access; disconnected inference is possible only after those prerequisites are available locally. The site itself is static and cannot host the model runtime.

1. Create and activate a virtual environment for your platform.
2. Install the dependencies while connected: `python -m pip install -r local/requirements.txt`.
3. Prepare a **copy** of your approved Qev checkpoint so its `model.json` field `model_id` is an absolute path to the matching local Qwen base directory. Preserve its adapter, `head.pt` and `temperature.json`. Do not modify the original files.
4. From this repository run `python local/try_qev.py --checkpoint <local-checkpoint-directory>` and open `http://127.0.0.1:7861` on the same machine. For a supported GPU use `--device cuda:0`.

The sample contexts summarize public incident reports, linked in the playground; their questions are retrospective research prompts. Qev was not used in the reported incidents. Outputs are experimental, may be inaccurate, and must not be used alone for safety-critical operating decisions. No measured latency, cost reduction, or production accuracy is claimed. Use only information appropriate for your environment.
