"""Verify required imports and real CPU/CUDA operations without model downloads.

CUDA is required by default. --allow-cpu enables CPU work but NEVER marks
the assignment's GPU requirement as passed.
"""

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import write_json  # noqa: E402

REQUIRED = {
    "pandas": "pandas",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "spacy": "spacy",
    "torch": "torch",
    "pyarrow": "pyarrow",
    "datasets": "datasets",
    "bs4": "beautifulsoup4",
    "langdetect": "langdetect",
}


def verification_exit_code(report: dict, allow_cpu: bool) -> int:
    if not report["imports_ok"] or not report["smoke_tests_ok"]:
        return 1
    return 0 if report["cuda_verified"] or allow_cpu else 1


def verify_environment(allow_cpu: bool = False) -> dict:
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "virtual_environment_active": sys.prefix != sys.base_prefix,
        "mode": "cpu_allowed" if allow_cpu else "cuda_required",
        "imports": {},
        "smoke_tests": {},
        "cuda_available": False,
        "cuda_verified": False,
        "gpu_devices": [],
    }
    modules = {}
    for module_name, distribution in REQUIRED.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            report["imports"][module_name] = {"ok": True, "version": version(distribution)}
        except Exception as error:
            report["imports"][module_name] = {"ok": False, "error": str(error)}
        print(f"Import {module_name}: {report['imports'][module_name]['ok']}", flush=True)

    def smoke_test(name, operation):
        try:
            operation()
            report["smoke_tests"][name] = {"ok": True}
        except Exception as error:
            report["smoke_tests"][name] = {"ok": False, "error": str(error)}

    if "pandas" in modules:

        def pandas_test():
            frame = modules["pandas"].DataFrame({"text": ["hello", "world"]})
            if frame.shape != (2, 1):
                raise AssertionError("Unexpected DataFrame shape")

        smoke_test("pandas_dataframe", pandas_test)
    if "sentence_transformers" in modules:

        def sentence_transformers_test():
            if not callable(modules["sentence_transformers"].SentenceTransformer):
                raise AssertionError("SentenceTransformer class is unavailable")

        smoke_test("sentence_transformers_api", sentence_transformers_test)
    if "spacy" in modules:

        def spacy_test():
            doc = modules["spacy"].blank("en")("Week one environment check.")
            if len(doc) != 5:
                raise AssertionError("spaCy tokenization failed")

        smoke_test("spacy_tokenization", spacy_test)
    if "chromadb" in modules:

        def chroma_test():
            settings = modules["chromadb"].Settings(anonymized_telemetry=False)
            client = modules["chromadb"].EphemeralClient(settings=settings)
            collection = client.create_collection("week1_environment_check")
            try:
                collection.add(ids=["doc1"], documents=["check"], embeddings=[[1.0, 0.0]])
                result = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)
                if result["ids"] != [["doc1"]]:
                    raise AssertionError("ChromaDB add/query failed")
            finally:
                client.delete_collection("week1_environment_check")

        smoke_test("chromadb_add_query", chroma_test)
    if "torch" in modules:
        torch = modules["torch"]

        def cpu_test():
            matrix = torch.eye(2)
            if not torch.equal(matrix @ matrix, matrix):
                raise AssertionError("PyTorch CPU matrix multiplication failed")

        smoke_test("torch_cpu_operation", cpu_test)
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            try:
                report["gpu_devices"] = [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ]
                matrix = torch.eye(2, device="cuda")
                if not torch.equal(matrix @ matrix, matrix):
                    raise AssertionError("CUDA matrix multiplication failed")
                torch.cuda.synchronize()
                report["cuda_verified"] = True
            except Exception as error:
                report["cuda_error"] = str(error)
        else:
            report["cuda_error"] = (
                "CUDA unavailable. A supported NVIDIA GPU, compatible driver, and CUDA-enabled "
                "PyTorch are required. CPU mode does not satisfy the assignment GPU requirement."
            )
    report["imports_ok"] = all(item["ok"] for item in report["imports"].values())
    report["smoke_tests_ok"] = (
        report["virtual_environment_active"]
        and len(report["smoke_tests"]) == 5
        and all(item["ok"] for item in report["smoke_tests"].values())
    )
    report["assignment_gpu_requirement_met"] = report["cuda_verified"]
    report["passed"] = verification_exit_code(report, allow_cpu) == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/environment_report.json")
    args = parser.parse_args()
    report = verify_environment(args.allow_cpu)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return verification_exit_code(report, args.allow_cpu)


if __name__ == "__main__":
    raise SystemExit(main())
