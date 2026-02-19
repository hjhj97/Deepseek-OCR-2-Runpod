import base64
import os
import re
import tempfile
import traceback
import urllib.parse
import urllib.request

import runpod

from run_dpsk_ocr2_pdf import init_model, run_ocr


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _safe_basename(filename: str) -> str:
    if not filename:
        return "output"
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
    return stem or "output"


def _parse_data_url(pdf_base64: str) -> str:
    if pdf_base64.startswith("data:"):
        parts = pdf_base64.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URL format for pdf_base64")
        return parts[1]
    return pdf_base64


def _load_pdf_from_url(pdf_url: str) -> bytes:
    req = urllib.request.Request(
        pdf_url,
        headers={
            "User-Agent": "DeepSeek-OCR2-RunPod-Serverless/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _get_pdf_bytes(job_input: dict) -> tuple[bytes, str]:
    pdf_base64 = job_input.get("pdf_base64")
    pdf_url = job_input.get("pdf_url")
    filename = job_input.get("filename", "input.pdf")

    if pdf_base64:
        normalized = _parse_data_url(pdf_base64)
        try:
            pdf_bytes = base64.b64decode(normalized, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid pdf_base64: {exc}") from exc
        return pdf_bytes, filename

    if pdf_url:
        pdf_bytes = _load_pdf_from_url(pdf_url)
        if filename == "input.pdf":
            url_path = urllib.parse.urlparse(pdf_url).path
            guessed = os.path.basename(url_path) or "input.pdf"
            filename = guessed
        return pdf_bytes, filename

    raise ValueError("Either input.pdf_base64 or input.pdf_url is required")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    try:
        job_input = job.get("input") or {}
        pdf_bytes, filename = _get_pdf_bytes(job_input)

        include_mmd_det_text = _as_bool(job_input.get("include_mmd_det_text"), default=False)
        include_layout_pdf_base64 = _as_bool(job_input.get("include_layout_pdf_base64"), default=False)
        include_output_files_base64 = _as_bool(job_input.get("include_output_files_base64"), default=False)

        with tempfile.TemporaryDirectory(prefix="deepseek_ocr2_") as tmpdir:
            base_name = _safe_basename(filename)
            result = run_ocr(pdf_bytes, tmpdir, base_name)

            response = {
                "status": "completed",
                "filename": filename,
                "total_pages": result["total_pages"],
                "processed_page_count": result.get("processed_page_count", result["total_pages"]),
                "processed_pages": result.get("processed_pages"),
                "mmd_text": _read_text(result["mmd_path"]),
            }

            if include_mmd_det_text:
                response["mmd_det_text"] = _read_text(result["mmd_det_path"])

            if include_layout_pdf_base64:
                response["layout_pdf_base64"] = _read_base64(result["layout_pdf_path"])

            if include_output_files_base64:
                response["mmd_base64"] = _read_base64(result["mmd_path"])
                response["mmd_det_base64"] = _read_base64(result["mmd_det_path"])

            return response

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if _as_bool(os.getenv("RUNPOD_PRELOAD_MODEL"), default=True):
    init_model()


runpod.serverless.start({"handler": handler})
