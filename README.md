# CATDA (Corpus-aware Automated Text-to-Graph Catalyst Discovery Agent)

**(Formerly: catdataextractor)**

## Quickstart: Usage and Environment

- **Models supported**: `openai_*`, `google_*`, `deepseek_*` (set the matching API key)
- **Primary scripts**:
  - `python -m CATDA.extract_main`: extract CatGraph and/or generate ML dataset
  - `python -m CATDA.tools.neo4j.neo4j_import`: import CatGraph JSON into Neo4j
  - `python -m CATDA.launch_gradio`: launch CatAgent UI for querying

### Environment variables

Set these before running commands (Windows PowerShell examples):

- Model provider API key (depending on `--model` you choose):
  - `OPENAI_API_KEY`, or `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY`
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

## Workflow

### 0) Preprocess: Convert PDFs to Markdown (recommended)

CATDA works best when articles are pre-converted to clean Markdown. We recommend using the programs that have at least or better ability than Azure OCR Markdown pipeline already implemented in `CATDA/PDF_TO_MD/PaperExtract.py`. 

- Recommended engine: `azuremarkdown` (Azure Document Intelligence → Markdown). You can also change to other (opensource) OCR engines that performs better. Also, we provided some simple alternatives, but from our early testing, they struggle to treat the complex layout for scientific documents.
- Alternatives (also available in `PaperExtract.py`):
  - `azure` (Azure layout JSON + our own text assembly)
  - `fitz` (text-based PDFs only, via PyMuPDF)
  - `paddle` (PaddleOCR for image-based PDFs)
  - `nougat` (math-heavy papers; CLI or API)

Setup Azure keys first in `PaperPreprocess/src/settings.py`:
- Set `azure_api_endpoint`, `azure_api_key`, and `azure_auto_load`

Batch conversion example (Python snippet, also main program in PDF_TO_MD/PaperExtract.py):
```python
from CATDA.PDF_TO_MD.PaperExtract import paper_parse
import os, glob

src_dir = r"D:\path\to\PDFs"
out_dir = r"D:\path\to\MD"
os.makedirs(out_dir, exist_ok=True)

for pdf_path in glob.glob(os.path.join(src_dir, "*.pdf")):
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_dir = os.path.join(out_dir, paper_name)
    paper_parse(pdf_path, save_dir, extraction_engine="azuremarkdown")

# Each paper's text is saved to: <save_dir>/txt/output.txt. You need to collect and move them into a folder of md for batch CATDA extraction
```

Notes:
- Downstream CATDA extraction expects your converted files under one root directory; pass `--file-ext .txt` when using the txt outputs (default is `.md`).
- Optional system/extras: `azure-ai-documentintelligence`, `pymupdf`, `opencv-python`, and (for `paddle`/`camelot`) additional native deps may be required.

### 1) Extract CatGraph and/or Generate ML Dataset

Command (run from the project root):
```bash
python -m CATDA.extract_main <input_path> \
  --output-dir <out_dir> \
  --file-ext .md \
  --mode both \
  --processes 4 \
  --feature-file CATDA/prompts/features_to_extract.txt
```

Key flags:
- `input_path`: file or directory of source texts
- `--output-dir`: where results are written (creates `graph/`, `dataset/`, `metadata/`)
- `--file-ext`: input extension filter (default `.md`)
- `--mode`: `extract` | `generate-ml-only` | `both` (default `both`)
- `--graph-pattern`: pattern for existing graphs when using `generate-ml-only` (default `*_output.json` under `<out_dir>/graph`)
- `--feature-file`: feature definitions for the ML dataset (default `CATDA/prompts/features_to_extract.txt`)

Outputs:
- CatGraph JSON: `<out_dir>/graph/<run_id>_output.json`
- ML dataset (TSV): `<out_dir>/dataset/<run_id>_dataset.tsv`
- Aux logs/metadata under `<out_dir>/metadata/` and temp artifacts

### 2) Import CatGraph JSON into Neo4j

Start Neo4j, then run:
```bash
python -m CATDA.tools.neo4j.neo4j_import <input_json_or_dir> \
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
python -m CATDA.launch_gradio \
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
python -m CATDA.extract_main <input_path> \
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

1) Extract CatGraph only
```bash
python -m CATDA.extract_main data/OCM_articles_MD --output-dir out_v1 --file-ext .md --mode extract --processes 4
```

2) Generate ML dataset from existing CatGraph
```bash
python -m CATDA.extract_main out_v1 --output-dir out_v1 --mode generate-ml-only --graph-pattern "*_output.json" --feature-file CATDA/prompts/features_to_extract.txt
```

3) Import to Neo4j
```bash
python -m CATDA.tools.neo4j.neo4j_import out_v1/graph --neo4j_user neo4j --neo4j_password "$NEO4J_PASSWORD" --clear
```

4) Launch CatAgent
```bash
python -m CATDA.launch_gradio --model google_gemini-2.5-pro --neo4j-password "$NEO4J_PASSWORD" --gradio-port 6810
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
