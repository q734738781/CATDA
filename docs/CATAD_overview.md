# CatData Extractor v3 – System Overview

*Prepared for catalysis-researchers exploring Large-Language-Model (LLM) applications*

---

## 1  Background & Motivation
Modern heterogeneous‐catalyst discovery relies on rapidly converting the ever-growing body of publications, patents and electronic-lab-notebook (ELN) records into structured knowledge that can be searched, reasoned over and ultimately used to guide new experiments.  CatData Extractor v3 transforms unstructured textual artefacts into three mutually-reinforcing artefacts:

1. **CatGraph** – a domain-specific knowledge graph capturing entities and relations relevant to catalysis experiments.
2. **ML Dataset** – instruction–response examples distilled from CatGraph, suitable for tuning LLMs on catalysis tasks.
3. **CatAgent** – a tool-augmented LLM that queries CatGraph (and auxiliary tools) to answer scientists' questions with verifiable evidence.

The following sections introduce the software components that implement this workflow.

---

## 2  CatGraph Extractor
*Implemented mainly in* `extract_main.py` *via* [`extract_catgraph`](../tools/cat_graph/catgraph_extractor.py)

### 2.1 Purpose
CatGraph Extractor ingests markdown, plain-text or PDF-converted text and emits:
* A JSON-serialised knowledge graph (`*_output.json`) following the Neo4j schema described in `neo4j_schema.md`.
* Rich metadata describing extraction success, model usage statistics and potential "missing-item" flags.

### 2.2 Processing Pipeline
1. **Task discovery** – Input paths are scanned (recursively when directories are supplied) for files matching a user-defined extension (`--file-ext`, default `.md`).
2. **Multiprocessing orchestration** – Work is parallelised across CPU cores; each worker calls `process_file_wrapper`.
3. **LLM‐driven extraction** – Within each worker an LLM (default Google Gemini 2.5 Pro, configurable) is initialised through `models.get_model`.  The heavy-lifting is delegated to `extract_catgraph`, which prompts the LLM to:
   * identify catalysis entities (Catalyst, Chemical, Condition, SynthesisStep, Measurement, etc.);
   * assign globally unique identifiers (UUID v4) and normalised field names;
   * output a JSON graph that conforms to Neo4j import format.
4. **Quality checks** – Post-extraction validation flags missing mandatory entities or relations, recording findings in the result JSON.
5. **Result materialisation** – 
   * `output_dir/graph/…`: graph JSON files
   * `output_dir/metadata/…`: per-file result summaries with usage telemetry (token counts, latency, etc.)

### 2.3 Key CLI Flags
| Flag | Description | Example |
|------|-------------|---------|
| `--mode extract` | run extraction only | default |
| `--processes 8` | #worker processes | speed-up large corpora |
| `--output-dir my_run` | destination folder | keeps runs isolated |
| `--model-temp 0.0` (via env var) | deterministic extraction | reproducibility |

---

## 3  ML Dataset Generator
*Implemented in* `extract_main.py` *via* [`generate_ml_dataset`](../tools/ml_dataset/generate_dataset.py)

### 3.1 Purpose
The generator converts each CatGraph JSON into one or more instruction-response pairs.  These pairs teach an LLM how to reason over catalysis-specific structures, enabling:
* few-shot prompting for property prediction;
* supervised fine-tuning (SFT) or reinforcement learning (RLHF) for catalytic-knowledge mastery.

### 3.2 Operation Modes
There are two entry points:

1. **Inline mode** (`--mode both`) – run immediately after a successful extraction.  Useful for end-to-end pipelines.
2. **Standalone mode** (`--mode generate-ml-only`) – scan an existing `output_dir/graph/` folder for `*_output.json` files and (re)generate dataset rows.  Facilitates iterative dataset curation without re-extracting graphs.

### 3.3 Dataset Schema (excerpt)
| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Natural-language query grounded in CatGraph context. |
| `input` | string | Subgraph/metadata required to answer the instruction. |
| `output` | string | Ground-truth answer extracted from the graph. |
| `run_id` | string | Links the row back to its originating extraction run. |

All rows are consolidated into Parquet/JSONL files ready for open-source frameworks such as HuggingFace `datasets` or `trl`.

---

## 4  CatAgent – Tool-Augmented Reasoning
*Implemented in* `service/agent_setup.py`

CatAgent is a LangChain-based **tool calling agent** that situates an LLM inside a rich tool ecosystem centred on CatGraph.  Upon receiving a researcher's query it can decide, autonomously, to:
* run a **Cypher query** via `GraphQueryTool` to retrieve experimental facts;
* inspect the **graph schema** (`GraphSchemaTool`) to understand available relations;
* resolve ambiguous chemical or catalyst **names** (`NameResolverTool`, `FieldNameResolverTool`);
* fetch **evidence passages** that justify an answer (`EvidenceFetcherTool`);
* perform **unit conversions** (`UnitConverterTool`);
* explore **multistep synthesis pathways** (`SynthesisPathExplorerTool`).

### 4.1 Prompt Design
The system prompt (see `prompts/agent_prompt.py`) enumerates tool descriptions in a structured JSON block.  LangChain's `create_tool_calling_agent` wrapper instructs the underlying LLM to output a `tool_call` whenever tool usage improves answer quality.  Intermediate results accumulate in the `agent_scratchpad`, enabling chain-of-thought style reasoning with full transparency.

### 4.2 Execution Modes
CatAgent can be deployed:
* **Interactively** – CLI or Jupyter notebooks for ad-hoc Q&A.
* **As a micro-service** – behind a FastAPI/Gradio endpoint for lab-assistant chatbots.
* **In batch** – automated literature triage, hypothesis scoring, etc.

---

## 5  Reproducibility & Extensibility
* **Configuration by CLI & ENV-vars** – model names, temperatures, Neo4j credentials and cache paths are parameterised.
* **Plug-and-play LLMs** – `models.get_model` currently supports OpenAI, DeepSeek and Google Gemini families; adding local models (e.g. Mistral, Llama 3) requires only an additional loader.
* **Schema evolution** – `neo4j_schema.md` documents versioned graph schemas; CatGraph extractor supports forward-compatible node/edge additions.

---

## 6  Suggested Citation
> *If you build upon CatData Extractor v3, please cite:*
>
> *Zhou et al.*, "Automated Knowledge-Graph Construction and LLM-Augmented Reasoning for Heterogeneous Catalysis", *Submitted to Journal of Catalysis Informatics*, 2024.

---

## 7  Acknowledgements
This work was funded by the **XYZ Catalysis Center** and leverages open-source software from the LangChain, Neo4j and HuggingFace communities. 