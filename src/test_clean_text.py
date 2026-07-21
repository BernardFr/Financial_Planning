#!/usr/local/bin/python3
"""Standalone validation script for utilities.clean_text.

Exercises whitespace trimming/collapsing and non-printable/non-ASCII character
handling (non-breaking spaces, smart quotes, tabs, newlines), plus the
non-string passthrough behavior.
"""

from utilities import clean_text

CASES = [
    # (description, input, expected)
    ("1 leading space", " Hello", "Hello"),
    ("multiple leading spaces", "   Hello", "Hello"),
    ("1 trailing space", "Hello ", "Hello"),
    ("multiple trailing spaces", "Hello   ", "Hello"),
    ("leading + trailing tabs and spaces", "\t  Hello World  \t", "Hello World"),
    ("multiple internal spaces, multiple locations", "Hello    World   Foo", "Hello World Foo"),
    ("internal tab", "Hello\tWorld", "Hello World"),
    ("single newline", "Hello\nWorld", "Hello World"),
    ("multiple newlines", "Hello\n\n\nWorld", "Hello World"),
    ("carriage return", "Hello\rWorld", "Hello World"),
    ("CRLF", "Hello\r\nWorld", "Hello World"),
    ("non-breaking space", "Hello\xa0World", "Hello World"),
    ("smart double quotes", "“Hello”", "Hello"),
    ("smart apostrophe", "don’t", "don t"),
    ("em dash", "Hello—World", "Hello World"),
    ("already clean", "Hello World", "Hello World"),
    ("empty string", "", ""),
    ("whitespace only", "   \t  ", ""),
    ("non-string int passes through", 42, 42),
    ("non-string None passes through", None, None),
    ("non-string float passes through", 3.14, 3.14),
]


def main() -> None:
    failures = 0
    for description, value, expected in CASES:
        actual = clean_text(value)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] {description}: clean_text({value!r}) = {actual!r} (expected {expected!r})")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        raise SystemExit(f"{failures} test case(s) failed")


if __name__ == "__main__":
    main()
