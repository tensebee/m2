#!/usr/bin/env python3
"""
gutendex_fetch.py — Fetch Project Gutenberg books via the Gutendex API.

Features:
- Search terms, topics, language filters
- Exclusions (case-insensitive against title/subjects/bookshelves/authors)
- Preferred format selection with broad matching (plain/html/epub/mobi/pdf)
- Optional ZIP allowance when only zipped formats exist
- Concurrent downloads with resume + retries
- Min-size skipping (to avoid tiny indices)
- DRY-RUN mode to preview matches without downloading
- Verbose diagnostics and end-of-run summary
- Writes manifest.json and manifest.csv with local paths
- Optional --strip-gutenberg to remove header/footer between
  *** START OF ... and *** END OF ... markers (plain text files)

Examples:
  python gutendex_fetch.py \
    --search "encyclopedia,encyclopaedia,britannica" \
    --exclude "juvenile,almanac" \
    --topic "reference,encyclopaedia" \
    --lang en \
    --format plain \
    --allow-zip \
    --strip-gutenberg \
    --out ./gutenberg_dl \
    --max-pages 10 \
    --concurrency 6 \
    --verbose
"""

import argparse, csv, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except Exception:
    print("Requires 'requests'. Install with: pip install requests", file=sys.stderr)
    raise

GUTENDEX_BASE = "https://gutendex.com/books"

FORMAT_PRIORITY = {
    "plain": ["text/plain; charset=utf-8", "text/plain"],
    "html":  ["text/html; charset=utf-8", "text/html"],
    "epub":  ["application/epub+zip"],
    "mobi":  ["application/x-mobipocket-ebook"],
    "pdf":   ["application/pdf"],
}

# Robust START/END patterns (case-insensitive, multiline)
START_PAT = re.compile(
    r"^\s*\*{3}\s*START OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK.*$", re.I | re.M
)
END_PAT = re.compile(
    r"^\s*\*{3}\s*END OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK.*$", re.I | re.M
)

def parse_args():
    p = argparse.ArgumentParser(description="Fetch Project Gutenberg texts via Gutendex.")
    p.add_argument("--search", type=str, default="", help="Comma-separated search terms (title/author).")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated exclusion terms (case-insensitive, checks title/subjects/bookshelves/authors).")
    p.add_argument("--topic", type=str, default="", help="Comma-separated topics (e.g., 'reference,encyclopaedia').")
    p.add_argument("--lang", type=str, default="en", help="Language codes (comma-separated, default: en).")
    p.add_argument("--format", type=str, choices=list(FORMAT_PRIORITY.keys()), default="plain", help="Preferred format: plain|html|epub|mobi|pdf")
    p.add_argument("--max-pages", type=int, default=5, help="Max pages to walk (pagination).")
    p.add_argument("--limit", type=int, default=0, help="Max books to download (0 = no limit).")
    p.add_argument("--out", type=str, default="./gutendex_out", help="Output directory.")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent downloads.")
    p.add_argument("--min-size", type=int, default=500, help="Skip files smaller than this many bytes.")
    p.add_argument("--retries", type=int, default=3, help="Retry attempts per file.")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    p.add_argument("--allow-zip", action="store_true", help="Allow downloading ZIP containers if selected format is only available zipped.")
    p.add_argument("--strip-gutenberg", action="store_true", help="For plaintext downloads, strip Gutenberg header/footer using *** START/END OF markers.")
    p.add_argument("--dry-run", action="store_true", help="Only list matches; don’t download files.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging for debugging.")
    return p.parse_args()

def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []

def build_query(params: Dict[str, str]) -> str:
    return GUTENDEX_BASE + "?" + "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items() if v)

def pick_format_url(formats: Dict[str, str], kind: str, allow_zip: bool, verbose: bool=False) -> Optional[Tuple[str, str]]:
    """Return (url, mime) for best match, or None."""
    # 1) exact priority list
    for mt in FORMAT_PRIORITY.get(kind, []):
        if mt in formats:
            return formats[mt], mt
    # 2) looser contains-match for the family (e.g., 'text/plain; charset=us-ascii')
    family = "text/plain" if kind == "plain" else ("text/html" if kind == "html" else None)
    if family:
        for mt, url in formats.items():
            if family in mt:
                return url, mt
    # 3) fallbacks
    for mt in ["text/plain; charset=utf-8", "text/plain", "text/html; charset=utf-8", "text/html", "application/epub+zip"]:
        if mt in formats:
            return formats[mt], mt
    # 4) zip containers if allowed (some entries only expose zips)
    if allow_zip:
        for mt in ["application/zip", "application/x-zip-compressed"]:
            if mt in formats:
                if verbose:
                    print(f"[VERBOSE] Using ZIP fallback: {formats[mt]} ({mt})")
                return formats[mt], mt
        # some items have misleading mime but zip URL
        for mt, url in formats.items():
            if url and url.endswith(".zip"):
                if verbose:
                    print(f"[VERBOSE] Using .zip by URL suffix: {url} ({mt})")
                return url, mt
    return None

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", (name or "").strip())
    name = re.sub(r"\s+", "_", name)
    return name[:200] if name else "unknown"

def excluded(book: Dict, needles: List[str]) -> bool:
    if not needles: return False
    hay = " ".join([
        book.get("title", "") or "",
        " ".join(book.get("subjects", []) or []),
        " ".join(book.get("bookshelves", []) or []),
        " ".join((a.get("name") or "") for a in (book.get("authors") or [])),
    ]).lower()
    return any(n.lower() in hay for n in needles)

def fetch_page(url: str, timeout: float) -> Dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def download_file(url: str, dest: Path, timeout: float, retries: int):
    # resume to .part then rename
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    last_err = None
    while attempt <= retries:
        attempt += 1
        headers = {}
        mode = "wb"
        existing = tmp.stat().st_size if tmp.exists() else 0
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:
                    tmp.rename(dest)
                    return True, None
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dest)
            return True, None
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2**attempt, 10))
    return False, last_err

def first_author_name(book: Dict) -> str:
    authors = book.get("authors") or []
    for a in authors:
        if isinstance(a, dict):
            name = (a.get("name") or "").strip()
            if name:
                return name
    return "unknown"

def strip_gutenberg_wrappers(text: str) -> str:
    """
    Return the substring between the *** START OF ... and *** END OF ... markers.
    If markers are not found, return the original text.
    """
    # Find first START and first END after it.
    start_match = START_PAT.search(text)
    end_match = END_PAT.search(text)

    if not start_match or not end_match:
        # Fallback heuristic: first line with "***" and "START" vs "***" and "END"
        lines = text.splitlines()
        s_idx, e_idx = None, None
        for i, ln in enumerate(lines):
            l = ln.strip().lower()
            if s_idx is None and l.startswith("***") and "start of" in l and "project gutenberg" in l:
                s_idx = i
            if e_idx is None and l.startswith("***") and "end of" in l and "project gutenberg" in l:
                e_idx = i
        if s_idx is not None and e_idx is not None and e_idx > s_idx:
            body = "\n".join(lines[s_idx + 1 : e_idx]).strip()
            return body + "\n"
        return text  # give up gracefully

    if end_match.start() <= start_match.end():
        return text  # pathological ordering; don't slice

    return text[start_match.end(): end_match.start()].strip() + "\n"

def main():
    args = parse_args()

    search_terms = split_csv_arg(args.search)
    topics = split_csv_arg(args.topic)
    excludes = split_csv_arg(args.exclude)
    langs = split_csv_arg(args.lang)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "files"; files_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = out_dir / "manifest.json"
    manifest_csv = out_dir / "manifest.csv"

    params = {}
    if search_terms: params["search"] = " ".join(search_terms)
    if topics: params["topic"] = " ".join(topics)
    if langs: params["languages"] = ",".join(langs)

    url = build_query(params) if params else GUTENDEX_BASE
    print(f"[INFO] Query: {url}")

    results = []
    pages = 0
    while url and pages < args.max_pages:
        pages += 1
        if args.verbose: print(f"[VERBOSE] GET {url}")
        try:
            data = fetch_page(url, timeout=args.timeout)
        except Exception as e:
            print(f"[WARN] Failed to fetch page {pages}: {e}", file=sys.stderr)
            break

        batch = data.get("results", []) or []
        print(f"[INFO] Page {pages}: {len(batch)} results")

        kept = [b for b in batch if not excluded(b, excludes)]
        print(f"[INFO] Page {pages}: {len(kept)} kept after exclusions")

        results.extend(kept)
        if args.limit and len(results) >= args.limit:
            results = results[:args.limit]
            break

        url = data.get("next")

    print(f"[INFO] Total kept: {len(results)}")

    # --- build download queue (robust) ---
    queue = []
    reasons = {"no_format": 0, "queued": 0, "downloaded": 0, "failed": 0, "skipped_small": 0}

    for b in results:
        book_id = b.get("id")
        raw_title = b.get("title") or f"book_{book_id}"
        title = (raw_title.strip() or f"book_{book_id}")
        author = first_author_name(b)

        formats = b.get("formats") or {}
        sel = pick_format_url(formats, args.format, args.allow_zip, args.verbose)
        if not sel:
            reasons["no_format"] += 1
            if args.verbose:
                print(f"[VERBOSE] No acceptable format for id={book_id} '{title}' (author={author})")
            continue

        href, mime = sel

        # choose extension using mime/url
        ext = ".txt"
        if ("text/html" in (mime or "")) or (href and href.endswith(".html")):
            ext = ".html"
        elif ("epub" in (mime or "")) or (href and href.endswith(".epub")):
            ext = ".epub"
        elif ("mobi" in (mime or "")) or (href and href.endswith(".mobi")):
            ext = ".mobi"
        elif ("pdf" in (mime or "")) or (href and href.endswith(".pdf")):
            ext = ".pdf"
        elif href and href.endswith(".zip"):
            ext = ".zip"

        fname = f"{sanitize_filename(author)}__{sanitize_filename(title)}__{book_id}{ext}"
        dest = files_dir / fname
        queue.append((href, dest, b))
        reasons["queued"] += 1
    # --- end queue build ---

    if args.limit and len(queue) > args.limit:
        queue = queue[:args.limit]

    print(f"[INFO] Download queue: {len(queue)}")

    downloaded = []
    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(download_file, url, dest, args.timeout, args.retries): (url, dest, meta)
                    for (url, dest, meta) in queue}
            for fut in as_completed(futs):
                url, dest, meta = futs[fut]
                ok, err = fut.result()
                if ok:
                    # Optional strip for plain text
                    if args.strip_gutenberg and dest.suffix.lower() == ".txt":
                        try:
                            raw = dest.read_text(encoding="utf-8", errors="ignore")
                            stripped = strip_gutenberg_wrappers(raw)
                            if stripped and stripped != raw:
                                dest.write_text(stripped, encoding="utf-8", errors="ignore")
                                if args.verbose:
                                    print(f"[CLEAN] Stripped Gutenberg header/footer: {dest.name}")
                        except Exception as e:
                            print(f"[WARN] Strip failed for {dest.name}: {e}", file=sys.stderr)

                    size = dest.stat().st_size if dest.exists() else 0
                    if size < args.min_size:
                        reasons["skipped_small"] += 1
                        if args.verbose:
                            print(f"[VERBOSE] Too small ({size} B): {dest.name}")
                        dest.unlink(missing_ok=True)
                        continue

                    print(f"[OK] {dest.name} ({size/1024:.1f} KiB)")
                    meta["_local_path"] = str(dest)
                    meta["_size_bytes"] = size
                    downloaded.append(meta)
                    reasons["downloaded"] += 1
                else:
                    reasons["failed"] += 1
                    print(f"[FAIL] {dest.name} <- {url}\n       {err}", file=sys.stderr)
    else:
        print("[DRY] Not downloading; writing manifest from queue only.")
        for _, dest, meta in queue:
            meta["_local_path"] = str(dest)
            meta["_size_bytes"] = None
            downloaded.append(meta)

    # Write manifest
    out_dir = Path(args.out)
    manifest_json = out_dir / "manifest.json"
    manifest_csv = out_dir / "manifest.csv"

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {manifest_json}")

    # CSV (guard None fields)
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "languages", "copyright",
            "download_count", "_size_bytes", "_local_path"
        ])
        writer.writeheader()
        for b in downloaded:
            writer.writerow({
                "id": b.get("id"),
                "title": (b.get("title") or f"book_{b.get('id')}"),
                "languages": ",".join(b.get("languages") or []),
                "copyright": b.get("copyright"),
                "download_count": b.get("download_count"),
                "_size_bytes": b.get("_size_bytes"),
                "_local_path": b.get("_local_path"),
            })
    print(f"[INFO] Wrote {manifest_csv}")

    print("[SUMMARY]",
          f"queued={reasons['queued']}",
          f"downloaded={reasons['downloaded']}",
          f"no_format={reasons['no_format']}",
          f"failed={reasons['failed']}",
          f"skipped_small={reasons['skipped_small']}", sep=" | ")
    print("[DONE]")

if __name__ == "__main__":
    main()
