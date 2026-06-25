from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


CODE_PATTERN = re.compile(r"^(I{1,3}|IV|V|VI|VII)[-.](\d+(?:[.-]\d+)*)\.?\s*(.*)$", re.IGNORECASE)
WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def load_definitions_from_docx(docx_path: str | Path, class_names: list[str] | tuple[str, ...]) -> dict[str, str]:
    class_set = set(class_names)
    definitions: dict[str, str] = {}
    pending_duplicates: list[tuple[str, str]] = []

    for paragraph in _read_docx_paragraphs(Path(docx_path)):
        match = CODE_PATTERN.match(paragraph)
        if not match:
            continue

        group = match.group(1).upper()
        code = f"{group}-{match.group(2).rstrip('.')}"
        definition = _clean_definition(match.group(3))
        normalized_code = _normalize_code(code, class_set)

        if normalized_code in definitions:
            pending_duplicates.append((normalized_code, definition))
            continue
        definitions[normalized_code] = definition

    for code, definition in pending_duplicates:
        target = _next_variant_code(code, class_set, definitions)
        if target is not None:
            definitions[target] = definition

    return definitions


def _read_docx_paragraphs(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        text = "".join((node.text or "") for node in paragraph.findall(".//w:t", WORD_NAMESPACE))
        text = " ".join(text.split())
        if text:
            paragraphs.append(text)
    return paragraphs


def _clean_definition(definition: str) -> str:
    return definition.strip().strip('“”"').strip()


def _normalize_code(code: str, class_set: set[str]) -> str:
    if code in class_set:
        return code

    group, rest = code.split("-", 1)
    candidates = [
        f"{group}-{rest.replace('.', '-')}",
        f"{group}-{rest.replace('-', '.')}",
    ]

    if "-" in rest and "." in rest:
        candidates.append(f"{group}-{rest.replace('-', '.', 1).replace('.', '-', 1)}")
        left, right = rest.rsplit(".", 1)
        candidates.append(f"{group}-{left.replace('-', '.')}-{right}")

    for candidate in candidates:
        if candidate in class_set:
            return candidate
    return code


def _next_variant_code(code: str, class_set: set[str], definitions: dict[str, str]) -> str | None:
    candidates = [f"{code}.1", f"{code}-1"]
    for candidate in candidates:
        if candidate in class_set and candidate not in definitions:
            return candidate
    return None
