"""Session-scoped document acquisition with source provenance."""
from __future__ import annotations

from data_collection_workflow.environment import get_env

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime, timezone

from .session_runtime import BudgetExceeded, ResumeMismatch

PARSER_VERSION = "universal-acquisition/3.11"
REQUIRED_LANGUAGES = {"eng", "osd", "fra", "spa", "por", "chi_sim"}


def _session(session_dir):
    if not session_dir:
        raise ValueError("session_dir is required; historical artifact discovery is forbidden")
    path = Path(session_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths(config=None):
    config = dict(config or {})
    path_file = Path(config.get("acquisition_paths_file") or Path(__file__).resolve().parents[2] / ".runtime/acquisition-paths.json")
    if path_file.exists():
        config = {**json.loads(path_file.read_text(encoding="utf-8-sig")), **config}
    for key, env in [("chromium_executable", "CHROMIUM_EXECUTABLE"), ("tesseract_executable", "TESSERACT_EXECUTABLE"), ("tessdata_dir", "TESSDATA_DIR")]:
        config[key] = get_env(env) or config.get(key)
    root = Path(__file__).resolve().parents[2]
    for key in ("chromium_executable", "tesseract_executable", "tesseract_working_directory", "ocr_staging_directory", "playwright_browsers_path"):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str(root / config[key])
    if config.get("playwright_browsers_path"):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = config["playwright_browsers_path"]
    return config


def _call(kind, payload, fn, recovery=False, source_target=None):
    try:
        from .session_runtime import get_runtime
    except ImportError:
        return fn()
    runtime = get_runtime()
    return runtime.call(kind, {"parser_version": PARSER_VERSION, **payload}, fn, recovery=recovery, source_target=source_target) if runtime else fn()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _store(session, body, suffix):
    relative = Path("acquisition") / (_hash(body) + suffix)
    destination = session / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return relative.as_posix()


def _mark_budget_exhausted(doc, error):
    """Preserve acquired evidence while identifying the operation left undone."""
    from .session_runtime import get_runtime
    adaptive = bool(getattr(getattr(get_runtime(), "ledger", None), "adaptive", False))
    if error.kind == "action_attempts" and adaptive:
        doc.update(acquisition_status="attempts_exhausted", fetch_status="failed",
                   fetch_error=str(error), acquisition_incomplete=False)
        doc.pop("budget_exhausted_kind", None)
        if "action_attempts_exhausted" not in doc.setdefault("quality_issues", []):
            doc["quality_issues"].append("action_attempts_exhausted")
        return doc
    doc.update(acquisition_status="budget_exhausted", budget_exhausted_kind=error.kind,
               acquisition_incomplete=True, fetch_error=str(error))
    issue = "budget_exhausted:" + str(error.kind or "unknown")
    if issue not in doc.setdefault("quality_issues", []):
        doc["quality_issues"].append(issue)
    return doc


def _deferred_fetch(url, source_id, session, error, response=None):
    # No HTTP response exists for a denied dispatch. Do not fabricate evidence
    # hashes, timestamps, HTTP statuses or an empty response artifact for it.
    doc = {"source_id": source_id, "url": url, "final_url": None,
           "retrieved_at": None, "is_live_fetched": False, "request_success": False,
           "content_readable": False, "clean_text": "", "tables": [],
           "locator_spans": [], "quality_issues": [], "ocr_words": [],
           "fetch_status": "budget_exhausted", "parse_status": "not_started",
           "parse_eligible": False, "raw_content_complete": False, "source_target_id": url,
           "target_data_status": "unavailable", "text_char_count": 0, "table_count": 0,
           "parser_version": PARSER_VERSION, "parser_used": PARSER_VERSION}
    if response is not None:
        # A redirect may have returned before its destination hit the cap.
        # Keep those bytes for audit, but they are not the destination's data.
        raw = base64.b64decode(response["body"])
        doc.update(raw_artifact_path=_store(session, raw, ".raw"), content_hash=_hash(raw),
                   final_url=response["final_url"], http_status_code=response["status_code"],
                   content_type=response["content_type"], is_live_fetched=True,
                   request_success=200 <= response["status_code"] < 400)
    _mark_budget_exhausted(doc, error)
    _store(session, json.dumps(doc, ensure_ascii=False).encode("utf-8"), ".json")
    return doc


def _browser(url, config):
    from .browser_transport import browser_fetch
    return browser_fetch(url, config)


def _ocr(image, config):
    executable = config.get("tesseract_executable") or "tesseract"
    languages = config.get("ocr_languages", "eng")
    if isinstance(languages, list):
        languages = "+".join(languages)
    available = _languages(config)
    missing = set(languages.split("+")) - available
    if missing:
        raise ValueError("unsupported OCR languages: " + ", ".join(sorted(missing)))
    staging = config.get("ocr_staging_directory")
    if staging:
        Path(staging).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as directory:
        image_path = Path(directory) / "page.png"
        image.save(image_path)
        cwd = config.get("tesseract_working_directory")
        image_argument = os.path.relpath(image_path, cwd) if cwd else str(image_path)
        command = [executable, image_argument, "stdout", "-l", languages]
        if config.get("tessdata_argument") or config.get("tessdata_dir"):
            command.extend(["--tessdata-dir", config.get("tessdata_argument") or config["tessdata_dir"]])
        command.append("tsv")
        result = _run_tesseract(command, config, timeout=90)
    words = []
    for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t"):
        if not row.get("text", "").strip():
            continue
        words.append({"text": row["text"], "confidence": float(row["conf"]), "bbox": [int(row[key]) for key in ("left", "top", "width", "height")], "line": [int(row[key]) for key in ("block_num", "par_num", "line_num")]})
    lines = []
    previous = None
    for word in words:
        if word["line"] != previous:
            lines.append([])
        lines[-1].append(word["text"])
        previous = word["line"]
    return {"text": "\n".join(" ".join(line) for line in lines), "words": words}


def _run_tesseract(command, config, timeout):
    env = dict(os.environ)
    env.pop("TESSDATA_PREFIX", None)
    return subprocess.run(command, capture_output=True, check=True, timeout=timeout, cwd=config.get("tesseract_working_directory"), env=env)


def _languages(config):
    command = [config.get("tesseract_executable") or "tesseract", "--list-langs"]
    if config.get("tessdata_argument") or config.get("tessdata_dir"):
        command.extend(["--tessdata-dir", config.get("tessdata_argument") or config["tessdata_dir"]])
    result = _run_tesseract(command, config, timeout=20)
    return set(result.stdout.decode("utf-8").splitlines()[1:])


def _html_table_structure(table, table_id):
    """Preserve physical DOM rows and cell geometry without fabricating text."""
    dom_rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    rows, cells, occupied = [], [], {}
    header_row_ids = []
    in_headers = True
    for row_id, row in enumerate(dom_rows):
        dom_cells = [cell for cell in row.find_all(["th", "td"])
                     if cell.find_parent("tr") is row and cell.find_parent("table") is table]
        values = [cell.get_text(" ", strip=True) for cell in dom_cells]
        rows.append(values)
        is_header_row = bool(dom_cells) and all(cell.name == "th" and cell.get("scope") != "row"
                                               for cell in dom_cells)
        if in_headers and is_header_row:
            header_row_ids.append(row_id)
        else:
            in_headers = False
        column = 0
        for dom_cell, quote in zip(dom_cells, values):
            while (row_id, column) in occupied:
                column += 1
            try:
                colspan = max(1, min(1000, int(dom_cell.get("colspan") or 1)))
                rowspan = int(dom_cell.get("rowspan") or 1)
                rowspan = len(dom_rows) - row_id if rowspan == 0 else max(1, rowspan)
                rowspan = min(rowspan, len(dom_rows) - row_id)
            except (TypeError, ValueError):
                colspan, rowspan = 1, 1
            cell = {"row_id": row_id, "column_start": column,
                    "column_end": column + colspan, "row_end": row_id + rowspan,
                    "quote": quote}
            cells.append(cell)
            for physical_row in range(row_id, row_id + rowspan):
                for physical_column in range(column, column + colspan):
                    occupied[physical_row, physical_column] = cell
            column += colspan
    # Retain the previous first-row convention for tables without explicit THs.
    if rows and not header_row_ids:
        header_row_ids = [0]
    for cell in cells:
        cell["is_header"] = cell["row_id"] in header_row_ids
    width = max((cell["column_end"] for cell in cells), default=0)
    column_labels = []
    for column in range(width):
        labels = []
        for cell in cells:
            if (cell["is_header"] and cell["column_start"] <= column < cell["column_end"]
                    and cell["quote"] and cell["quote"] not in labels):
                labels.append(cell["quote"])
        column_labels.append(" / ".join(labels))
    return {"table_id": table_id, "headers": rows[0] if rows else [],
            "header_row_ids": header_row_ids,
            "rows": [row for row_id, row in enumerate(rows) if row_id not in header_row_ids],
            "row_ids": [row_id for row_id in range(len(rows)) if row_id not in header_row_ids],
            "column_labels": column_labels, "cell_spans": cells}, rows



def _decode_response_text(body, content_type):
    """Require valid text; replacement-decoded binary is never evidence."""
    charset = re.search(r"charset\s*=\s*['\"]?([\w-]+)", content_type, re.I)
    encoding = charset.group(1) if charset else ("utf-16" if body.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    try:
        text = body.decode(encoding)
    except (UnicodeError, LookupError):
        return None
    controls = sum(ord(char) < 32 and char not in "\t\r\n\f" for char in text)
    if controls:
        # Extracted prose can retain sparse source formatting controls. Keep
        # their exact positions: deleting them could concatenate numeric tokens.
        declared_text = content_type.split(";", 1)[0].strip().lower().startswith("text/")
        if "\x00" in text or not declared_text or controls * 100 > len(text):
            return None
    return text


def _json_response(text, content_type):
    """Honor declared JSON; sniff other bodies only after validating the value."""
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in {"text/markdown", "text/x-markdown"}:
        return False
    if "json" in mime:
        return True
    if not text.lstrip().startswith(("{", "[")):
        return False
    try:
        return isinstance(json.loads(text), (dict, list))
    except (ValueError, TypeError):
        return False


def _image_response(body, content_type):
    return (content_type.lower().startswith("image/") or
            body.startswith((b"\x89PNG\r\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*")) or
            body[:4] == b"RIFF" and body[8:12] == b"WEBP")


def _cookie_access_gate(text):
    """Recognize an entire access instruction, never a notice within content.

    Cookie notices, quoted instructions and short reports remain readable. HTML
    callers omit site chrome only for this check; stored text and locators keep
    their original representation.
    """
    normalized = " ".join(str(text or "").split())
    english = (
        r"(?:cookies?\s+(?:must be enabled|are required)[.!:]?\s+)?"
        r"(?:please\s+)?enable\s+(?:cookies?(?:\s+and\s+javascript)?|javascript\s+and\s+cookies?)"
        r"(?:\s+(?:for|in)\s+.{1,160}?)?\s+(?:and\s+)?"
        r"(?:reload\s+(?:this|the)\s+page\s+to\s+continue|to\s+continue)[.!]?")
    french = (
        r"(?:veuillez\s+activer|activez)\s+(?:les\s+)?"
        r"(?:cookies(?:\s+et\s+javascript)?|javascript\s+et\s+(?:les\s+)?cookies)"
        r"\s+(?:et\s+recharger\s+(?:cette|la)\s+page\s+)?pour\s+continuer[.!]?")
    return any(re.fullmatch(pattern, normalized, re.I) for pattern in (english, french))


def parse_response(body, *, url, source_id, session_dir, content_type="text/plain", status_code=200, final_url=None, config=None, recovery=False, raw_content_complete=True):
    session = _session(session_dir)
    config = _paths(config)
    body = bytes(body)
    doc = {"source_id": source_id, "url": url, "final_url": final_url or url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "content_hash": _hash(body), "raw_artifact_path": _store(session, body, ".raw"), "parser_version": PARSER_VERSION, "parser_used": PARSER_VERSION, "content_type": content_type, "http_status_code": status_code, "request_success": 200 <= status_code < 300, "tables": [], "locator_spans": [], "quality_issues": [], "ocr_words": [], "is_live_fetched": True, "fetch_status": "success" if 200 <= status_code < 300 else "failed"}
    eligible = bool(body) and raw_content_complete and 200 <= status_code < 300
    doc.update(raw_content_complete=bool(raw_content_complete), parse_eligible=eligible,
               source_target_id=config.get("_source_target_id", url))
    invalid_response = ("truncated_response" if not raw_content_complete else
                        "empty_response" if not body and 200 <= status_code < 300 else
                        "redirect_response" if 300 <= status_code < 400 else
                        "http_error" if not 200 <= status_code < 300 else None)
    parts = []
    def append(text, **locator):
        if not text:
            return
        start = sum(len(part) + 1 for part in parts)
        parts.append(text)
        span = {"char_start": start, "char_end": start + len(text), **locator}
        doc["locator_spans"].append(span)
        return span
    text = _decode_response_text(body, content_type)
    parse_error = None
    gate_text = None
    try:
        if not eligible:
            doc["quality_issues"].append(invalid_response)
        elif body.startswith(b"%PDF") or ("pdf" in content_type and text is None):
            import pypdfium2 as pdfium
            doc["document_type"] = "pdf"
            pdf = pdfium.PdfDocument(body)
            try:
                for index in range(len(pdf)):
                    page = pdf[index]
                    try:
                        page_quality = []
                        page_method = "native_text"
                        textpage = page.get_textpage()
                        page_text = textpage.get_text_range()
                        textpage.close()
                        if config.get("force_ocr") or len(re.sub(r"\s", "", page_text)) < 20:
                            def recognize_page():
                                bitmap = page.render(scale=float(config.get("ocr_render_scale", 3)))
                                try:
                                    return _ocr(bitmap.to_pil(), config)
                                finally:
                                    bitmap.close()
                            try:
                                output = _call("ocr", {"document_hash": doc["content_hash"], "page": index + 1, "render_scale": config.get("ocr_render_scale", 3), "languages": config.get("ocr_languages", "eng")}, recognize_page, recovery)
                            except BudgetExceeded as exc:
                                _mark_budget_exhausted(doc, exc)
                                doc.setdefault("unprocessed_pages", []).append(index + 1)
                                # Keep native text even when OCR was needed to supplement it.
                                append(page_text.strip(), page=index + 1)
                                # Other pages can still contain native text or cached OCR.
                                continue
                            page_method = "ocr"
                            page_text = output["text"]
                            for word in output["words"]:
                                word["page"] = index + 1
                            doc["ocr_words"].extend(output["words"])
                            confidences = [word["confidence"] for word in output["words"]]
                            if not confidences or min(confidences) < float(config.get("ocr_min_confidence", 80)):
                                doc["quality_issues"].append("low_ocr_confidence_candidate")
                                page_quality.append("low_ocr_confidence_candidate")
                        append(page_text.strip(), page=index + 1, extraction_method=page_method, quality_issues=page_quality)
                    finally:
                        page.close()
            finally:
                pdf.close()
        elif _image_response(body, content_type):
            from PIL import Image
            doc["document_type"] = "image"
            with Image.open(io.BytesIO(body)) as picture:
                try:
                    output = _call("ocr", {"document_hash": doc["content_hash"], "page": 1,
                        "languages": config.get("ocr_languages", "eng")}, lambda: _ocr(picture, config), recovery)
                except BudgetExceeded as exc:
                    _mark_budget_exhausted(doc, exc)
                    doc["unprocessed_pages"] = [1]
                else:
                    doc["ocr_words"] = [{**word, "page": 1} for word in output["words"]]
                    confidence = [word["confidence"] for word in output["words"]]
                    issues = []
                    if not confidence or min(confidence) < float(config.get("ocr_min_confidence", 80)):
                        issues.append("low_ocr_confidence_candidate")
                        doc["quality_issues"].extend(issues)
                    append(output["text"], page=1, extraction_method="ocr", quality_issues=issues)
        elif ("spreadsheet" in content_type or "excel" in content_type or
              url.lower().split("?")[0].endswith((".xlsx", ".xls"))):
            # Retain the real workbook for a supported parser; binary bytes are
            # never text evidence merely because UTF-8 replacement can decode them.
            doc["document_type"] = "spreadsheet"
            doc["quality_issues"].append("unsupported_spreadsheet_format")
        elif text is None:
            doc["document_type"] = "binary"
            doc["quality_issues"].append("unsupported_binary_format")
        elif _json_response(text, content_type):
            doc["document_type"] = "json"
            append(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
        elif any(kind in content_type for kind in ("csv", "tab-separated")) or url.lower().split("?")[0].endswith((".csv", ".tsv")):
            doc["document_type"] = "csv"
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if "tab-separated" in content_type or url.lower().split("?")[0].endswith(".tsv") else ","))
            if rows:
                doc["tables"].append({"table_id": "table_1", "headers": rows[0], "rows": rows[1:]})
                for index, row in enumerate(rows):
                    append(" | ".join(row), table_id="table_1", row_id=index)
        elif "html" in content_type or text.lstrip().lower().startswith(("<html", "<!doctype")):
            from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
            doc["document_type"] = "html"
            soup = BeautifulSoup(text, "html.parser")
            has_script = bool(soup.find("script"))
            for node in soup(["script", "style", "noscript"]):
                node.decompose()
            gate_scope = soup.body or soup
            gate_text = " ".join(str(node).strip() for node in gate_scope.find_all(string=True)
                if not isinstance(node, (Comment, Doctype))
                and not node.find_parent(["head", "nav", "footer"])
                and not (node.find_parent("header") and not node.find_parent(["main", "article"])))
            from .resource_discovery import html_resource_links
            main = soup.find("main") or soup.find(attrs={"role": "main"})
            page_heading = main.find(["h1", "h2"]) if main is not None else soup.find("h1")
            if page_heading is not None and page_heading.find_parent(["nav", "footer"]):
                page_heading = None
            doc["metadata"] = {
                "page_title": soup.title.get_text(" ", strip=True) if soup.title else "",
                "page_heading": page_heading.get_text(" ", strip=True) if page_heading is not None else "",
                "outbound_links": html_resource_links(
                    soup, source_url=doc["final_url"], content_hash=doc["content_hash"]),
            }
            headings = {id(node): (int(node.name[1:]), f"heading_{index}")
                        for index, node in enumerate(soup.find_all(re.compile(r"^h[1-6]$")), 1)}
            table_ids = {id(table): f"table_{index}" for index, table in enumerate(soup.find_all("table"), 1)}
            for node in soup.descendants:
                if getattr(node, "name", None) == "table":
                    table_id = table_ids[id(node)]
                    parsed_table, rows = _html_table_structure(node, table_id)
                    caption = node.caption.get_text(" ", strip=True) if node.caption else None
                    parsed_table["caption"] = caption
                    doc["tables"].append(parsed_table)
                    if caption:
                        append(caption, table_id=table_id, table_part="caption")
                    cells_by_row = {}
                    for cell in parsed_table["cell_spans"]:
                        cells_by_row.setdefault(cell["row_id"], []).append(cell)
                    for row_index, row in enumerate(rows):
                        span = append(" | ".join(row), table_id=table_id, row_id=row_index,
                                      table_part="header" if row_index in parsed_table["header_row_ids"] else "row")
                        if span is not None:
                            offset = span["char_start"]
                            for cell in cells_by_row.get(row_index, []):
                                cell.update(char_start=offset, char_end=offset + len(cell["quote"]))
                                offset += len(cell["quote"]) + 3
                elif isinstance(node, NavigableString) and not isinstance(node, Comment):
                    table = node.find_parent("table")
                    if table is None:
                        heading = node.find_parent(re.compile(r"^h[1-6]$"))
                        location = {}
                        if heading is not None:
                            level, section = headings[id(heading)]
                            location = {"role": "heading", "heading_level": level, "section_id": section}
                        append(str(node).strip(), **location)
                    elif not node.find_parent(["tr", "caption"]):
                        append(str(node).strip(), table_id=table_ids[id(table)], table_part="note")
            visible = "\n".join(parts)
            from .page_status import is_dynamic_shell
            doc["is_shell"] = is_dynamic_shell({**doc, "clean_text": visible}, has_script=has_script)
        else:
            doc["document_type"] = "text"
            append(text.strip())
            gate_text = text
    except (BudgetExceeded, ResumeMismatch):
        raise
    except Exception as exc:
        # Failed parsing retains the received bytes and any spans already read.
        parse_error = type(exc).__name__ + ": " + str(exc)
        doc.update(parse_error=parse_error, fetch_error=parse_error)
        doc["quality_issues"].append("parse_error")
    doc["clean_text"] = "\n".join(parts)
    doc["text_hash"] = _hash(doc["clean_text"].encode("utf-8"))
    doc["text_artifact_path"] = _store(session, doc["clean_text"].encode("utf-8"), ".txt")
    from .page_status import page_failure_reason
    failure = page_failure_reason(doc)
    if not failure and _cookie_access_gate(gate_text):
        failure = "blocked"
    status = invalid_response or failure or ("shell" if doc.get("is_shell") else "readable" if doc["clean_text"].strip() else "unreadable")
    if failure and failure not in doc["quality_issues"]:
        doc["quality_issues"].append(failure)
    doc.update(acquisition_status=status, content_readable=status == "readable", target_data_status="unassessed" if status == "readable" else "unavailable", parse_status="parsed" if status == "readable" else "failed", text_char_count=len(doc["clean_text"]), table_count=len(doc["tables"]))
    if failure == "blocked":
        doc.update(parse_eligible=False, fetch_status="failed")
    elif doc.get("is_shell"):
        doc["parse_eligible"] = False
    if doc.get("acquisition_incomplete"):
        doc.update(acquisition_status="budget_exhausted", parse_status="parsed_partial" if doc["content_readable"] else "failed")
    elif parse_error:
        doc.update(acquisition_status="parse_error", parse_status="parsed_partial" if doc["content_readable"] else "failed")
    elif any(issue in doc["quality_issues"] for issue in ("unsupported_spreadsheet_format", "unsupported_binary_format")):
        doc.update(acquisition_status="unsupported_format", parse_status="unsupported_format", parse_eligible=False,
                   content_readable=False, target_data_status="unavailable")
    _store(session, json.dumps(doc, ensure_ascii=False).encode("utf-8"), ".json")
    return doc


def _browser_interface_configuration(document):
    """Identify captured site-interface definitions, retaining their audit artifact.

    These structures describe survey placement and controls. They are not the
    statistical JSON/CSV responses loaded by the page's data presentation.
    """
    if document.get("document_type") != "json":
        return False
    try:
        payload = json.loads(document.get("clean_text") or "")
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    intercept = payload.get("InterceptDefinition")
    if isinstance(intercept, dict) and isinstance(intercept.get("ActionSets"), dict):
        actions = intercept["ActionSets"].values()
        if any(isinstance(action, dict) and
               (action.get("CreativeType") == "FeedbackButton" or
                (isinstance(action.get("Target"), dict) and action["Target"].get("Type") == "Survey"))
               for action in actions):
            return True
    creative = payload.get("CreativeDefinition")
    if isinstance(creative, dict) and isinstance(creative.get("Options"), dict):
        if any(isinstance(option, dict) and isinstance(option.get("LookAndFeel"), dict)
               for option in creative["Options"].values()):
            return True
    intercepts = payload.get("Intercepts")
    return bool(isinstance(intercepts, list) and intercepts and
                all(isinstance(item, dict) and "InterceptID" in item and isinstance(item.get("Decision"), dict)
                    and ("SurveyID" in item or "Creative" in item["Decision"])
                    for item in intercepts))


class PartialResponseError(RuntimeError):
    """Retain received bytes without caching an incomplete operation as success."""
    def __init__(self, message, response_payload, *, retryable=False):
        super().__init__(message)
        self.response_payload = response_payload
        self.retryable = retryable


def _request_response(url, config):
    import requests
    with requests.get(url, timeout=config.get("timeout_seconds", 30), stream=True, allow_redirects=False) as response:
        payload = {"body": "", "content_type": response.headers.get("content-type", "text/plain"),
                   "status_code": response.status_code, "final_url": response.url,
                   "redirect_location": response.headers.get("location"), "raw_content_complete": True}
        chunks, size = [], 0
        try:
            for chunk in response.iter_content(65536):
                if size + len(chunk) > int(config.get("max_bytes", 20_000_000)):
                    payload.update(body=base64.b64encode(b"".join(chunks)).decode(), raw_content_complete=False)
                    raise PartialResponseError("response exceeds max_bytes", payload)
                chunks.append(chunk)
                size += len(chunk)
        except requests.RequestException as exc:
            payload.update(body=base64.b64encode(b"".join(chunks)).decode(), raw_content_complete=False)
            raise PartialResponseError(str(exc), payload, retryable=True) from exc
        payload["body"] = base64.b64encode(b"".join(chunks)).decode()
        length = response.headers.get("content-length")
        if length and str(length).isdigit() and not response.headers.get("content-encoding") and size != int(length):
            payload["raw_content_complete"] = False
            raise PartialResponseError("incomplete Content-Length", payload, retryable=True)
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise PartialResponseError("temporary HTTP " + str(response.status_code), payload, retryable=True)
        return payload


def _failed_document(url, source_id, error, session, response=None):
    doc = {"source_id": source_id, "source_target_id": url, "url": url, "final_url": None,
           "retrieved_at": None, "is_live_fetched": False, "request_success": False,
           "content_readable": False, "clean_text": "", "tables": [], "locator_spans": [],
           "quality_issues": ["request_error"], "ocr_words": [], "fetch_status": "failed",
           "parse_status": "not_started", "parse_eligible": False, "raw_content_complete": False,
           "target_data_status": "unavailable", "text_char_count": 0, "table_count": 0,
           "parser_version": PARSER_VERSION, "parser_used": PARSER_VERSION,
           "acquisition_status": "request_error", "fetch_error": str(error)}
    if response is not None:
        raw = base64.b64decode(response["body"])
        doc.update(raw_artifact_path=_store(session, raw, ".raw"), content_hash=_hash(raw),
                   final_url=response["final_url"], http_status_code=response["status_code"],
                   content_type=response["content_type"], is_live_fetched=True,
                   raw_content_complete=bool(response.get("raw_content_complete", True)))
    return doc


def acquire_document(url: str, *, source_id: str, session_dir, config=None, recovery=False, priority_context=None):
    """Acquire one logical target; redirects and alternative transports share it."""
    session = _session(session_dir)
    from .session_runtime import get_runtime
    from .acquisition_budget import canonical_source_target
    runtime = get_runtime()
    if get_env("PIPELINE_MODE") == "evidence" and runtime is None:
        raise RuntimeError("evidence acquisition requires an active session runtime")
    if runtime is not None and getattr(runtime, "session_dir", session) != session:
        raise ValueError("acquisition session_dir does not match active runtime session")
    config = _paths(config)
    source_target = canonical_source_target(config.get("_source_target_id") or url)
    config["_source_target_id"] = source_target
    adaptive = bool(getattr(getattr(runtime, "ledger", None), "adaptive", False))
    strategy = config.get("acquisition_strategy", "native")
    if strategy not in {"native", "browser"}:
        raise ValueError("unknown acquisition_strategy: " + str(strategy))
    import requests
    from urllib.parse import urljoin
    current_url, response = url, None
    attempts = []
    transport_error = None
    if strategy == "native":
        try:
            for redirect in range(6):
                priority = priority_context or {}
                verified_priority = (priority.get("source_id") == source_id and
                    priority.get("verified_url") == current_url and
                    all(priority.get(key) is True for key in ("source_identity_verified", "must_fetch", "task_fit_verified")))
                fetch_kind = "fetch_priority" if verified_priority else "fetch_ordinary"
                for attempt in range(2 if adaptive else 1):
                    try:
                        response = dict(_call(fetch_kind, {"url": current_url, "max_bytes": config.get("max_bytes", 20_000_000)},
                            lambda: _request_response(current_url, config), recovery, source_target=source_target))
                        break
                    except (requests.RequestException, PartialResponseError) as exc:
                        partial = getattr(exc, "response_payload", None)
                        if partial is not None:
                            response = dict(partial)
                        audit = {"provider": "native_requests", "url": current_url, "success": False, "error": str(exc)}
                        if partial is not None:
                            raw = base64.b64decode(partial["body"])
                            audit.update(raw_artifact_path=_store(session, raw, ".raw"), content_hash=_hash(raw),
                                         http_status_code=partial["status_code"])
                        attempts.append(audit)
                        if attempt + 1 >= (2 if adaptive else 1) or not getattr(exc, "retryable", True):
                            raise
                location = response.get("redirect_location")
                if response["status_code"] not in {301, 302, 303, 307, 308} or not location:
                    break
                if redirect == 5:
                    raise ValueError("redirect limit exceeded (five hops)")
                current_url = urljoin(current_url, location)
        except BudgetExceeded as exc:
            return _deferred_fetch(url, source_id, session, exc, response)
        except (requests.RequestException, PartialResponseError, ValueError) as exc:
            transport_error = exc
        if transport_error is not None:
            doc = _failed_document(url, source_id, transport_error, session, response)
        else:
            parsed_response = {k: v for k, v in response.items() if k != "redirect_location"}
            doc = parse_response(base64.b64decode(parsed_response.pop("body")), url=url, source_id=source_id,
                session_dir=session_dir, config=config, recovery=recovery, **parsed_response)
    else:
        doc = _failed_document(url, source_id, "native transport not requested", session)
    doc.update(source_target_id=source_target, provider_attempts=attempts, fetch_provider="native_requests")
    if strategy == "browser" or doc["acquisition_status"] in {"shell", "blocked"} or doc.get("http_status_code") in {401, 403}:
        partial_browser = None
        try:
            def render():
                rendered = _browser(current_url, config)
                if rendered.get("budget_exhausted_kind") or rendered.get("request_errors") or rendered.get("unsupported_browser_contexts"):
                    raise PartialResponseError("browser acquisition incomplete", rendered)
                return rendered
            rendered = _call("browser_navigation" if adaptive else "browser_fetch",
                {"url": current_url, "render_wait_ms": config.get("render_wait_ms", 300)}, render,
                recovery, source_target=source_target)
        except PartialResponseError as exc:
            rendered = exc.response_payload
            if rendered.get("budget_exhausted_kind"):
                partial_browser = BudgetExceeded(str(exc), kind=rendered["budget_exhausted_kind"])
        except BudgetExceeded as exc:
            _mark_budget_exhausted(doc, exc)
            _store(session, json.dumps(doc, ensure_ascii=False).encode("utf-8"), ".json")
            return doc
        except ResumeMismatch:
            raise
        except Exception as exc:
            doc.update(acquisition_status="browser_error", fetch_error=type(exc).__name__ + ": " + str(exc))
            doc["quality_issues"].append("browser_error")
            _store(session, json.dumps(doc, ensure_ascii=False).encode("utf-8"), ".json")
            return doc
        rendered = dict(rendered)
        captured = rendered.pop("browser_responses", [])
        rendered.pop("budget_exhausted_kind", None)
        unsupported = rendered.pop("unsupported_browser_contexts", [])
        request_errors = rendered.pop("request_errors", [])
        original = doc
        doc = parse_response(base64.b64decode(rendered.pop("body")), url=url, source_id=source_id, session_dir=session_dir, config=config, recovery=recovery, **rendered)
        if original.get("raw_artifact_path"):
            doc["response_artifact_path"] = original["raw_artifact_path"]
            doc["response_content_hash"] = original["content_hash"]
        doc["provider_attempts"] = attempts
        doc["metadata"] = {**doc.get("metadata", {}), "browser_request_errors": request_errors,
                           "unsupported_browser_contexts": unsupported}
        doc["fetch_provider"] = "chromium"
        doc["browser_responses"] = []
        for response_index, captured_response in enumerate(captured, 1):
            child = parse_response(base64.b64decode(captured_response["body"]), url=captured_response["url"], source_id=source_id, session_dir=session_dir, content_type=captured_response["content_type"], status_code=captured_response["status_code"], config=config)
            if _browser_interface_configuration(child):
                child["evidence_exclusion_reason"] = "site_interface_configuration"
            doc["browser_responses"].append(child)
            if (child["content_readable"] and not child.get("evidence_exclusion_reason")
                    and doc.get("acquisition_status") in {"readable", "shell"}):
                offset = len(doc["clean_text"]) + (1 if doc["clean_text"] else 0)
                doc["clean_text"] += ("\n" if doc["clean_text"] else "") + child["clean_text"]
                table_ids = {table["table_id"]: f"browser_response_{response_index}__{table['table_id']}" for table in child["tables"]}
                for span in child["locator_spans"]:
                    translated = {**span, "char_start": span["char_start"] + offset, "char_end": span["char_end"] + offset,
                                  "response_url": child["url"], "response_content_hash": child["content_hash"]}
                    if span.get("table_id") in table_ids:
                        translated["table_id"] = table_ids[span["table_id"]]
                    doc["locator_spans"].append(translated)
                for table in child["tables"]:
                    translated = {**table, "table_id": table_ids[table["table_id"]]}
                    if table.get("cell_spans"):
                        translated["cell_spans"] = [{**cell, "char_start": cell["char_start"] + offset,
                                                     "char_end": cell["char_end"] + offset} for cell in table["cell_spans"]]
                    doc["tables"].append(translated)
                if child.get("document_type") == "json":
                    doc["locator_spans"].append({"char_start": offset, "char_end": offset + len(child["clean_text"]),
                        "structured_data_kind": "json_document", "response_url": child["url"], "response_content_hash": child["content_hash"]})
                doc.update(content_readable=True, acquisition_status="readable", target_data_status="unassessed", parse_status="parsed")
        doc["text_hash"] = _hash(doc["clean_text"].encode("utf-8"))
        doc["text_artifact_path"] = _store(session, doc["clean_text"].encode("utf-8"), ".txt")
        doc["text_char_count"] = len(doc["clean_text"])
        doc["table_count"] = len(doc["tables"])
        if partial_browser:
            _mark_budget_exhausted(doc, partial_browser)
        elif unsupported or request_errors:
            doc["acquisition_incomplete"] = True
            doc["quality_issues"].append("browser_resources_incomplete")
    _store(session, json.dumps(doc, ensure_ascii=False).encode("utf-8"), ".json")
    return doc


def preflight_acquisition(config, session_dir):
    """Actual loopback JS navigation and PDFium + Tesseract scan, before paid calls."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from PIL import Image, ImageDraw, ImageFont
    config = _paths(config)
    session = _session(session_dir)
    version_result = _run_tesseract([config.get("tesseract_executable") or "tesseract", "--version"], config, timeout=20)
    version = (version_result.stdout + version_result.stderr).decode()
    if not re.search(r"tesseract\s+v?5\.", version, re.I):
        raise RuntimeError("Tesseract 5 is required")
    missing = REQUIRED_LANGUAGES - _languages(config)
    if missing:
        raise RuntimeError("Missing OCR languages: " + ", ".join(sorted(missing)))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b'<div id="result">Loading...</div><script>setTimeout(()=>document.getElementById("result").textContent="DYNAMIC PREFLIGHT 731",100)</script>')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rendered = _browser(f"http://127.0.0.1:{server.server_port}/", config)
    finally:
        server.shutdown()
        server.server_close()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(base64.b64decode(rendered["body"]), "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    if "DYNAMIC PREFLIGHT 731" not in soup.get_text():
        raise RuntimeError("Chromium dynamic-page preflight failed")
    image = Image.new("RGB", (1400, 260), "white")
    font = ImageFont.truetype(config.get("preflight_font", "arial.ttf" if os.name == "nt" else "DejaVuSans.ttf"), 64)
    ImageDraw.Draw(image).text((40, 70), "SCAN PREFLIGHT 731", fill="black", font=font)
    pdf_buffer = io.BytesIO()
    image.save(pdf_buffer, format="PDF", resolution=150)
    # Local readiness is deliberately outside paid-operation accounting.
    doc = parse_response(pdf_buffer.getvalue(), url="local:preflight-scan", source_id="preflight", session_dir=session / "preflight", config={**config, "ocr_languages": "eng"})
    if "SCAN PREFLIGHT 731" not in doc["clean_text"] or not doc["ocr_words"]:
        raise RuntimeError("PDFium/Tesseract scanned-page preflight failed")
    report = {"ready": True, "dynamic_page": True, "scanned_page": True, "languages": sorted(_languages(config)), "parser_version": PARSER_VERSION, "tesseract_version": version.splitlines()[0], "checked_at": datetime.now(timezone.utc).isoformat()}
    (session / "acquisition-preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
