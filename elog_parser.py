"""Parse an ELOG (ELOGD) ``export.xml`` dump into a Polars DataFrame.

The ELOG exporter writes entry text verbatim, so the file is *not* well-formed
XML: ``TEXT`` bodies routinely contain raw ``&`` and ``<`` characters that break
strict parsers (lxml/ElementTree). The structure is, however, extremely regular:

    <ENTRY>
        <MID>...</MID>            # one per line, mandatory
        <DATE>...</DATE>
        <ENCODING>...</ENCODING>
        <Author>...</Author>
        <System>...</System>
        <Flags>...</Flags>
        <Subject>...</Subject>
        <IN_REPLY_TO>...          # optional, may repeat
        <REPLY_TO>...             # optional, may repeat
        <ATTACHMENT>...           # optional, may repeat (comma-separated values)
        <TEXT>... multi-line ...</TEXT>   # always last
    </ENTRY>

so we split on entry boundaries and pull each field out with anchored regexes.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import polars as pl

# DATE looks like: "Fri, 26 Jun 2026 08:00:00 +0200"  (RFC-2822-ish)
_DATE_FMT = "%a, %d %b %Y %H:%M:%S %z"

# Single-line scalar tags -> output column name.
_SCALAR_TAGS = {
    "MID": "mid",
    "DATE": "date",
    "ENCODING": "encoding",
    "Author": "author",
    "System": "system",
    "Flags": "flags",
    "Subject": "subject",
}

_scalar_re = {tag: re.compile(rf"<{tag}>(.*?)</{tag}>") for tag in _SCALAR_TAGS}
# TEXT is multi-line; grab everything up to the last </TEXT> in the entry.
_text_re = re.compile(r"<TEXT>(.*)</TEXT>", re.DOTALL)
_attach_re = re.compile(r"<ATTACHMENT>(.*?)</ATTACHMENT>")
_inreply_re = re.compile(r"<IN_REPLY_TO>(.*?)</IN_REPLY_TO>")
_reply_re = re.compile(r"<REPLY_TO>(.*?)</REPLY_TO>")

# Output column -> dtype. Drives the DataFrame schema (and documents the result).
_SCHEMA: dict[str, pl.DataType] = {
    "mid": pl.Int64,
    "date": pl.Utf8,
    "encoding": pl.Utf8,
    "author": pl.Utf8,
    "system": pl.Utf8,
    "flags": pl.Utf8,
    "subject": pl.Utf8,
    "text": pl.Utf8,
    "attachments": pl.List(pl.Utf8),
    "in_reply_to": pl.List(pl.Int64),
    "reply_to": pl.List(pl.Int64),
}


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    return int(value) if value else None


def _int_list(matches: list[str]) -> list[int]:
    """Flatten tag matches (each possibly comma-separated) into a list of ints."""
    out: list[int] = []
    for m in matches:
        for part in m.split(","):
            v = _to_int(part)
            if v is not None:
                out.append(v)
    return out


def _parse_entry(chunk: str) -> dict:
    """Extract a single ELOG ``<ENTRY>`` chunk into a record dict."""
    rec: dict = {}
    for tag, col in _SCALAR_TAGS.items():
        m = _scalar_re[tag].search(chunk)
        rec[col] = m.group(1) if m else None

    text_m = _text_re.search(chunk)
    rec["text"] = html.unescape(text_m.group(1)) if text_m else None

    # ATTACHMENT may repeat and each tag may hold comma-separated filenames.
    attachments: list[str] = []
    for a in _attach_re.findall(chunk):
        attachments.extend(p.strip() for p in a.split(",") if p.strip())
    rec["attachments"] = attachments

    # IN_REPLY_TO / REPLY_TO may repeat and may be comma-separated lists of MIDs.
    rec["in_reply_to"] = _int_list(_inreply_re.findall(chunk))
    rec["reply_to"] = _int_list(_reply_re.findall(chunk))

    # Tidy scalar string fields (trim + unescape XML entities).
    rec["mid"] = _to_int(rec["mid"])
    for col in ("encoding", "author", "system", "flags", "subject"):
        rec[col] = html.unescape(rec[col].strip()) if rec[col] else None

    return rec


def parse_export(path: str | Path) -> pl.DataFrame:
    """Parse an ELOG ``export.xml`` file into a Polars DataFrame.

    Returns one row per logbook entry, sorted by ``mid``, with a tz-aware
    ``date`` column (UTC). See ``_SCHEMA`` for the full column layout.
    """
    # ELOG declares ISO-8859-1; decode as latin-1 so e.g. "\xb0C" survives.
    raw = Path(path).read_bytes().decode("latin-1")

    # Each entry's content lives between <ENTRY> and </ENTRY>. Splitting on the
    # closing tag and keeping anything after a <MID> is robust to the optional
    # fields and to the verbatim (unescaped) body text.
    records = [_parse_entry(chunk) for chunk in raw.split("</ENTRY>") if "<MID>" in chunk]

    df = pl.DataFrame(records, schema=_SCHEMA)

    # Parse the RFC-2822-style timestamp into a tz-aware Datetime.
    return df.with_columns(
        pl.col("date")
        .str.strptime(pl.Datetime(time_unit="us", time_zone="UTC"), _DATE_FMT, strict=False)
        .alias("date")
    ).sort("mid")
