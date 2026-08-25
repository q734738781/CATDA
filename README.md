# CATDA (Corpus-aware Automated Text-to-Graph Catalyst Discovery Agent)

**Accepted Paper in ACS Catalysis**

DOI link: https://doi.org/10.1021/acscatal.5c06431 (Available after proof)

## Quickstart: Usage and Environment
**Important Note:** In the age of LLM, you can use agents (e.g. codex) to read this readme and set up environment interactively. Even fix bugs for version change in dependent package!
- **Models supported**: `openai_*`, `google_*`, `deepseek_*`, `openrouter_*` (set the matching API key)
- **Primary scripts**:
  - `python -m CATDA.extract_main`: extract CatGraph and/or generate ML dataset
  - `python -m CATDA.tools.neo4j.neo4j_import`: import CatGraph JSON into Neo4j
  - `python -m CATDA.launch_gradio`: launch CatAgent UI for querying

### Environment variables

Set these before running commands (Windows PowerShell examples):

- Model provider API key (depending on `--model` you choose):
  - `OPENAI_API_KEY`, or `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY`, or `OPENROUTER_API_KEY`
- Neo4j connection:
  - `NEO4J_URI` (default `neo4j://localhost:7687`)
  - `NEO4J_USER` (default `neo4j`)
  - `NEO4J_PASSWORD` (required)
- Optional regex mapping files for resolvers:
  - `NAME_RESOLVER_REGEX_MAP` (path to JSON)
  - `FIELD_RESOLVER_REGEX_MAP` (path to JSON)

Examples:
```powershell
$env:GOOGLE_API_KEY = "<your_key>"
$env:NEO4J_PASSWORD = "<neo4j_password>"
# Optional
$env:NAME_RESOLVER_REGEX_MAP = "D:\configs\name_regex.json"
$env:FIELD_RESOLVER_REGEX_MAP = "D:\configs\field_regex.json"
```

OpenRouter note:
- Use model names like `openrouter_openai/gpt-5.4` with `OPENROUTER_API_KEY`.
- OpenRouter reasoning models can use `--reasoning-effort minimal|low|medium|high|xhigh`.
- Use `requirements-pinned.txt` if you need to reproduce the LangChain 1.x validated environment.

## Workflow

### 0) Preprocess: Convert PDFs to Markdown (client only)
**Note: there seems to be an issue that PaddleOCR client may have confict with current conversation agent's similar keyword fetch. Please test and consider if you should create a new environment seperately for preprocess only.**
Use the client script `pdf_preprocess/PaddleOCR_vl_pdf2md_client.py` for PDF → Markdown preprocessing.

This repository only provides the client. Deploy a PaddleOCR‑VL server separately using the official guide:
`paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html`

Example (batch directory):
```bash
python pdf_preprocess/PaddleOCR_vl_pdf2md_client.py <input_pdf_or_dir> \
  -o <output_dir> \
  --vllm-url http://127.0.0.1:8118/v1 \
  --pdf-batch-size 4 \
  --vl-rec-max-concurrency 128 \
  --clean-html-tables


```

Notes:
- If you wish paddleocr to read chart figures for you, add --expand-charts
- Output is written as `<output_dir>/<relative_path>/<stem>/<stem>.md` with an `img/` folder.
- Clean HTML tables are good enough for LLM extraction in recent tests (see: `https://www.improvingagents.com/blog/best-input-data-format-for-llms`).
- Downstream CATDA extraction expects files under one root directory; pass `--file-ext .txt` if your outputs are `.txt` (default is `.md`).
### 1) Extract CatGraph and/or Generate ML Dataset

Command (run from the project root):
```bash
export PYTHONPATH=.
python extract_main.py <input_path> \
  --output-dir <out_dir> \
  --file-ext .md \
  --mode both \
  --processes 4 \
  --model openrouter_openai/gpt-5.4 \
  --ml-model openrouter_openai/gpt-5.4 \
  --reasoning-effort medium \
  --feature-file prompts/features_to_extract.txt
```

Key flags:
- `input_path`: file or directory of source texts
- `--output-dir`: where results are written (creates `graph/`, `dataset/`, `metadata/`)
- `--file-ext`: input extension filter (default `.md`)
- `--mode`: `extract` | `generate-ml-only` | `both` (default `both`)
- `--model`: extraction model in `provider_model` format
- `--ml-model`: model used to generate projected feature rows in `--mode both` or `generate-ml-only`
- `--reasoning-effort`: optional reasoning effort for supported OpenRouter models
- `--graph-pattern`: pattern for existing graphs when using `generate-ml-only` (default `*_output.json` under `<out_dir>/graph`)
- `--feature-file`: feature definitions for the ML dataset (default `prompts/features_to_extract.txt` when running from this directory)

Outputs:
- CatGraph JSON: `<out_dir>/graph/<run_id>_output.json`
- ML dataset (TSV): `<out_dir>/dataset/<run_id>_dataset.tsv`
- Aux logs/metadata under `<out_dir>/metadata/` and temp artifacts

### 2) Import CatGraph JSON into Neo4j

Start Neo4j, then run:
```bash
export PYTHONPATH=.
python -m tools/neo4j/neo4j_import <input_json_or_dir> \
  --neo4j_uri neo4j://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_password "$NEO4J_PASSWORD" \
  --clear
```

Notes:
- `--clear` wipes the DB before the first import; omit for incremental loads
- For a single file, `--paper_name` overrides the Paper node name (otherwise derived from filename)

### 3) Launch CatAgent (Gradio UI)

```bash
export PYTHONPATH=.
python -m launch_gradio.py \
  --model google_gemini-2.5-pro \
  --neo4j-password "$NEO4J_PASSWORD" \
  --gradio-port 6810 \
  --listen-all
```

Optional overrides:
- `--name-regex-map`, `--field-regex-map` to supply resolver JSONs (see examples below)

#### Optional Regex Mapping Files (Name/Field Resolvers)
If provided, exact/regex matches take precedence over vector search.

Object-map format:
```json
{
  "mfi|mobil five|mordenite": "MFI",
  "silic.*": "silica"
}
```

List format:
```json
[
  {"pattern": "conv(ersion)?\\s*rate", "name": "result_conversion_pct"},
  {"pattern": "yield", "key": "result_yield_pct"}
]
```

## Customization

### Projected features for the ML dataset

- Edit the feature spec in `CATDA/prompts/features_to_extract.txt` (PX-ISOMERIZATION example) or `CATDA/prompts/features_to_extract_OCM.txt` (OCM example). The expected format is a simple `|`-separated description list with two sections: “Catalyst properties” and “Testing condition/metrics”. Because it will be provided as a text block, so format is free as LLM will correctly handle it.
- Point the extractor at a custom file with `--feature-file <path>` when running `CATDA.extract_main`.

Typical call with a custom feature file:
```bash
export PYTHONPATH=.
python -m extract_main.py <input_path> \
  --output-dir out \
  --mode generate-ml-only \
  --feature-file D:/configs/my_features.txt
```

### When CatGraph does not meet task-specific requirements

Adjust the extraction prompts used to build CatGraph:
- File: `CATDA/prompts/extract_prompts.py`
  - Core synthesis prompt: variable `synthesis_graph_prompt`
  - Post-check for synthesis: `synthesis_missing_check_prompt`
  - The same file also contains prompts for characterization/testing phases

Guidance:
- Preserve the overall output schema keys expected by the importer and downstream tools (`nodes`, `edges`, field names like `synthesis_input`, `synthesis_output`, etc.)
- Tighten or relax rules in the prompt sections (e.g., condition capture, property handling) to suit your domain
- Keep JSON-only answers enforced in the prompt to simplify parsing

## Concrete Examples

1) Extract CatGraph and projected feature table for the included CN102039159A example
```bash
export OPENROUTER_API_KEY="<your_openrouter_key>"
python extract_main.py examples/CN102039159A.md \
  --output-dir examples/CN102039159A_test_run \
  --file-ext .md \
  --mode both \
  --processes 1 \
  --model openrouter_openai/gpt-5.4 \
  --ml-model openrouter_openai/gpt-5.4 \
  --reasoning-effort medium \
  --feature-file prompts/features_to_extract.txt
```

The published-reference extraction for this example is stored in `examples/CN102039159A_ref/`.
See `examples/CN102039159A_usage.md` for the example layout and quick comparison script.

2) Extract CatGraph only
```bash
export PYTHONPATH=.;python -m extract_main.py data/OCM_articles_MD --output-dir out_v1 --file-ext .md --mode extract --processes 4
```

3) Generate ML dataset from existing CatGraph
```bash
export PYTHONPATH=.;python extract_main.py out_v1 --output-dir out_v1 --mode generate-ml-only --graph-pattern "*_output.json" --feature-file prompts/features_to_extract.txt
```

4) Import to Neo4j
```bash
export PYTHONPATH=.;python -m tools.neo4j.neo4j_import out_v1/graph --neo4j_user neo4j --neo4j_password "$NEO4J_PASSWORD" --clear
```

5) Launch CatAgent
```bash
export PYTHONPATH=.;python -m launch_gradio --model google_gemini-2.5-pro --neo4j-password "$NEO4J_PASSWORD" --gradio-port 6810
```

## Installation

1) Clone
```bash
git clone <repository-url>
cd <repository-root>
```

2) Environment
- Conda (recommended):
```bash
conda create -n catda python=3.10 -y
conda activate catda
pip install -r CATDA/requirements.txt
```
- venv (alternative):
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or: source venv/bin/activate
pip install -r CATDA/requirements.txt
```

## Key Features

- LLM-powered extraction and reasoning
- CatGraph representation for catalyst synthesis/testing
- Full-document parsing (beyond abstracts)
- CatAgent for interactive, grounded queries over Neo4j graphs
- ML-ready dataset generation from CatGraph

## Project Structure (at a glance)

```
CATDA/
├── .cache/          # Cache directory
├── agentic_tools/   # Tools for the agentic components
├── docs/            # Documentation
├── examples/        # Example files and use cases
├── models/          # Model wrappers and utilities
├── prompts/         # LLM prompts and feature specs
├── service/         # Service-related code
├── tools/           # Utilities (incl. neo4j importer)
├── ui/              # Gradio UI
├── extract_main.py  # Extraction + dataset generation entrypoint
├── launch_gradio.py # Launch the Gradio UI
├── requirements.txt # Python dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please open issues/PRs for improvements.

We are currently working ahead for a multimodel version of CATDA, CATDA-MM. If you are interested in the project and want to collaborate, you may contact the corresponding author of the paper or me for further work.
