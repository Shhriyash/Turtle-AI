"""Run Zusatzinfo chunking/classification repeatedly to inspect output variance.

This script uses the existing project functions and model client:
- parse_doc.extract_process_data (includes Zusatzinfo image extraction + chunking)
- llm_gpt_classify_zusatinfo_chunks
- write_classification_results

It runs the same input multiple times in sequence and writes JSON outputs for each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from KIPPSFlow.config import Config, LLMConfig
from KIPPSFlow.llm.init.model import GPT
from KIPPSFlow.llm.zusatzinfos.classify import llm_gpt_classify_zusatinfo_chunks
from KIPPSFlow.logger import logger, set_run_id
from KIPPSFlow.parsing.parse_doc import extract_process_data
from KIPPSFlow.parsing.parseddata import HauptProzessContent, ProzessContent
from KIPPSFlow.zusatzinfo.classification_result import write_classification_results

PACKAGED_PROMPTS_DIR = REPO_ROOT / "src" / "KIPPSFlow" / "files" / "prompts"
REQUIRED_PROMPTS = ("zusatzinfos_chunking.txt", "zusatzinfos_classify.txt")


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _resolve_process(
    content: HauptProzessContent | ProzessContent, process_key: str | None
) -> ProzessContent:
    if isinstance(content, ProzessContent):
        actual_key = f"{content.prozess_nr}{content.variant}"
        if process_key and process_key != actual_key:
            raise ValueError(
                f"Provided --process-key '{process_key}' does not match parsed standalone process '{actual_key}'."
            )
        return content

    assert isinstance(content, HauptProzessContent)
    available = sorted(content.prozess_contents.keys())
    if process_key:
        if process_key not in content.prozess_contents:
            raise ValueError(
                "Process key not found in PDF. "
                f"Requested: '{process_key}'. Available: {available}"
            )
        return content.prozess_contents[process_key]

    if len(content.prozess_contents) == 1:
        return next(iter(content.prozess_contents.values()))

    raise ValueError(
        "PDF contains multiple processes; pass --process-key. "
        f"Available: {available}"
    )


def _ensure_prompt_files(prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PROMPTS:
        target = prompt_dir / name
        if target.exists():
            continue
        source = PACKAGED_PROMPTS_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Missing packaged prompt file: {source}")
        shutil.copy2(source, target)


def _build_conf(
    app_dir: Path,
    prompt_dir: Path | None,
    endpoint: str,
    api_key: str,
    api_version: str,
    model: str,
) -> Config:
    llm_conf = LLMConfig(
        azure_openai_endpoint=endpoint,
        azure_openai_api_key=api_key,
        azure_openai_api_ver=api_version,
        azure_openai_model=model,
    )
    conf = Config(
        azure_conn_str="local-no-storage",
        llm_conf=llm_conf,
        app_dir=app_dir,
    )
    if prompt_dir is not None:
        conf.prompt_dir = prompt_dir
    _ensure_prompt_files(conf.prompt_dir)
    conf.tmp_dir.mkdir(parents=True, exist_ok=True)
    conf.output_dir.mkdir(parents=True, exist_ok=True)
    return conf


def _result_file_name(prozess: ProzessContent) -> str:
    variant_suff = f"_{prozess.variant}" if prozess.variant else ""
    return f"{prozess.prozess_nr}{variant_suff}.json"


def run_variation_check(
    *,
    ist_pdf: Path,
    process_key: str | None,
    runs: int,
    output_root: Path,
    conf: Config,
) -> dict:
    gpt_client = GPT(conf)
    summary_runs: list[dict] = []

    for i in range(1, runs + 1):
        run_name = f"run_{i:02d}"
        set_run_id(run_name)
        run_out_dir = output_root / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)

        # Keep each run isolated; extract_process_data writes PNGs under tmp_dir.
        if conf.tmp_dir.exists():
            shutil.rmtree(conf.tmp_dir)
        conf.tmp_dir.mkdir(parents=True, exist_ok=True)

        run_context = SimpleNamespace(
            conf=conf,
            gpt_client=gpt_client,
            run_out_dir=run_out_dir,
        )

        logger.info("Starting %s", run_name)
        parsed_content = extract_process_data(run_context, ist_pdf)
        prozess = _resolve_process(parsed_content, process_key)

        _dump_json(
            run_out_dir / "input_snapshot.json",
            {
                "prozess_nr": prozess.prozess_nr,
                "variant": prozess.variant,
                "chunk_count": len(prozess.zusatzinformationen),
            },
        )
        _dump_json(run_out_dir / "chunks.json", prozess.zusatzinformationen)

        classification = llm_gpt_classify_zusatinfo_chunks(run_context, prozess)
        _dump_json(run_out_dir / "classification_raw.json", classification)
        write_classification_results(run_context, prozess, classification)

        result_path = (
            run_out_dir / "zusatzinfo_klassifizierung" / _result_file_name(prozess)
        )
        if not result_path.exists():
            raise FileNotFoundError(f"Expected classification output not found: {result_path}")

        result_bytes = result_path.read_bytes()
        result_sha = hashlib.sha256(result_bytes).hexdigest()
        summary_runs.append(
            {
                "run": run_name,
                "result_file": str(result_path.relative_to(output_root)),
                "sha256": result_sha,
                "size_bytes": len(result_bytes),
            }
        )
        logger.info("Finished %s (sha256=%s)", run_name, result_sha)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ist_pdf": str(ist_pdf),
        "runs": summary_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Zusatzinfo classification repeatedly for the same input."
    )
    parser.add_argument(
        "--ist-pdf",
        type=Path,
        required=True,
        help="Path to IST PDF input.",
    )
    parser.add_argument(
        "--process-key",
        type=str,
        default=None,
        help=(
            "Optional Prozess key for multi-process PDFs, e.g. '1.02.55.610' "
            "or '1.02.55.610_neo'."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of sequential runs (default: 5).",
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=Path("local_app"),
        help="Working app dir for tmp/output (default: local_app).",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help="Optional prompt dir. Defaults to <app-dir>/conf/prompts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root output dir for this experiment. "
            "Default: <app-dir>/output/zusatzinfo_variation_<timestamp>."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        help="Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_KEY", ""),
        help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        help="Azure OpenAI API version.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2"),
        help="Azure OpenAI deployment/model name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.ist_pdf.exists():
        print(f"IST PDF not found: {args.ist_pdf}", file=sys.stderr)
        return 2
    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2
    if not args.endpoint or not args.api_key:
        print(
            "Missing endpoint/api-key. Pass --endpoint and --api-key "
            "or set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        args.app_dir / "output" / f"zusatzinfo_variation_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    conf = _build_conf(
        app_dir=args.app_dir,
        prompt_dir=args.prompt_dir,
        endpoint=args.endpoint,
        api_key=args.api_key,
        api_version=args.api_version,
        model=args.model,
    )

    summary = run_variation_check(
        ist_pdf=args.ist_pdf.resolve(),
        process_key=args.process_key,
        runs=args.runs,
        output_root=output_root.resolve(),
        conf=conf,
    )
    _dump_json(output_root / "summary.json", summary)

    print(f"Done. Outputs written to: {output_root}")
    print(f"Summary: {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())