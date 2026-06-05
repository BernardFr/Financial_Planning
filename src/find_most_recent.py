#!/usr/local/bin/python3
import sys
import re
from pathlib import Path
from datetime import date as dt, datetime
from tempfile import TemporaryDirectory
from logger import logger

DEFAULT_DATE_FORMAT = "%Y_%m_%d"


def _datetime_format_to_regex(date_format: str) -> str:
    token_map = {
        "%Y": r"\d{4}",
        "%y": r"\d{2}",
        "%m": r"\d{1,2}",
        "%d": r"\d{1,2}",
    }

    parts: list[str] = []
    idx = 0
    while idx < len(date_format):
        if idx + 1 < len(date_format) and date_format[idx] == "%":
            token = date_format[idx : idx + 2]
            if token in token_map:
                parts.append(token_map[token])
                idx += 2
                continue

        ch = date_format[idx]
        if ch in "_.-":
            # Allow these separators only inside the date segment.
            parts.append(r"[-_.]")
        else:
            parts.append(re.escape(ch))
        idx += 1

    return "".join(parts)


def find_most_recent( directory_path: str, filename_prefix: str, date_format: str = DEFAULT_DATE_FORMAT) -> tuple[Path | None, str]:
    directory = Path(directory_path)
    date_regex = _datetime_format_to_regex(date_format)
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_({date_regex})")

    best = None
    best_date = None
    date_str = ""

    for f in directory.iterdir():
        # Make sure f is a file
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            matched_date = m.group(1)
            try:
                normalized_date = re.sub(r"[-_.]", "_", matched_date)
                normalized_format = re.sub(r"[-_.]", "_", date_format)
                parsed_date = datetime.strptime(normalized_date, normalized_format).date()
            except ValueError:
                continue
            if best_date is None or parsed_date > best_date:
                best_date = parsed_date
                best = f
                date_str = parsed_date.strftime("%Y_%m_%d")

    # if we don't find any matching files, look for files with the prefix but no date
    if best is None:
        print(f"No files found with prefix {filename_prefix!r} and date in {directory_path}")
        print("Looking for files with prefix but no date.", file=sys.stderr)
        for f in directory.iterdir():
            if f.name.startswith(filename_prefix):
                best = f
                # date_str is today's date in YYYY_MM_DD format
                date_str = dt.today().strftime("%Y_%m_%d")
                break

    return best, date_str   


def _run_main_tests() -> int:
    """Run basic end-to-end filename/date-format scenarios."""
    scenarios = [
        {
            "name": "dot separator, 2-digit year",
            "filename": "life_plan_2.5.26.xlsx",
            "prefix": "life_plan",
            "date_format": "%m.%d.%y",
            "expected": "2026_02_05",
        },
        {
            "name": "dash separator, 4-digit year",
            "filename": "life_plan_02-12-2026.xlsx",
            "prefix": "life_plan",
            "date_format": "%m-%d-%Y",
            "expected": "2026_02_12",
        },
        {
            "name": "underscore separator",
            "filename": "life_plan_2026_2_9.xlsx",
            "prefix": "life_plan",
            "date_format": "%Y_%m_%d",
            "expected": "2026_02_09",
        },
        {
            "name": "format uses underscore, filename uses dots",
            "filename": "life_plan_2026.11.1.xlsx",
            "prefix": "life_plan",
            "date_format": "%Y_%m_%d",
            "expected": "2026_11_01",
        },
        {
            "name": "format uses dots, filename uses dashes",
            "filename": "life_plan_2026-12-31.xlsx",
            "prefix": "life_plan",
            "date_format": "%Y.%m.%d",
            "expected": "2026_12_31",
        },
        {
            "name": "month/day/year format without separators",
            "filename": "life_plan_052326.xlsx",
            "prefix": "life_plan",
            "date_format": "%m%d%y",
            "expected": "2026_05_23",
        },
        {
            "name": "month/day/year format without separators - and extra underscore in filename",
            "filename": "life_plan_052326_1233.xlsx",
            "prefix": "life_plan",
            "date_format": "%m%d%y",
            "expected": "2026_05_23",
        },
    ]

    passed = 0
    for scenario in scenarios:
        with TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / scenario["filename"]
            test_file.touch()
            result, date_str = find_most_recent(
                tmp_dir,
                scenario["prefix"],
                scenario["date_format"],
            )
            ok = result is not None and result.name == scenario["filename"] and date_str == scenario["expected"]
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {scenario['name']}: file={scenario['filename']} -> {date_str}")
            if ok:
                passed += 1

    print(f"\nSelf-test summary: {passed}/{len(scenarios)} passed")
    return 0 if passed == len(scenarios) else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(_run_main_tests())

    if len(sys.argv) not in [3, 4]:
        print("Usage: find_most_recent.py <directory_path> <filename_prefix> [date_format]")
        print("       find_most_recent.py --self-test")
        sys.exit(1)

    date_format = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_DATE_FORMAT
    result, date_str = find_most_recent(sys.argv[1], sys.argv[2], date_format)
    if result:
        print(f"Most recent file: {result}\nDate: {date_str}")
    else:
        print(f"No matching files found.", file=sys.stderr)
        sys.exit(1)