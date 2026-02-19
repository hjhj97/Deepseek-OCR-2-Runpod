#!/usr/bin/env python3
import base64
from pathlib import Path

# Edit this list in source code (no CLI args needed).
PDF_FILES = [
    "test_02.pdf",
    # "test_01.pdf",
    # "/Users/hajuheon/Task/sample/test_04.pdf",
]


def convert_file(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only .pdf is supported: {input_path}")

    output_path = input_path.with_suffix(input_path.suffix + ".base64")
    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    output_path.write_text(encoded, encoding="ascii")
    print(f"[OK] {input_path} -> {output_path} ({output_path.stat().st_size} bytes)")


def main() -> None:
    if not PDF_FILES:
        raise ValueError("Set at least one file path in PDF_FILES")

    for file_path in PDF_FILES:
        convert_file(Path(file_path).expanduser().resolve())


if __name__ == "__main__":
    main()
