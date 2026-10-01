"""Create a separate LaTeX copy without \\hl wrappers, preserving their text."""

import argparse
from pathlib import Path
import re


HIGHLIGHT = re.compile(r"\\hl(?![A-Za-z])\s*\{")
# Ignore comments and escaped braces while counting nested groups.
TOKEN = re.compile(r"%[^\r\n]*|\\(?:[A-Za-z]+|[^\r\n])|[{}]")


def remove_highlights(text):
    count = 0
    while match := HIGHLIGHT.search(text):
        depth = 1
        for token in TOKEN.finditer(text, match.end()):
            if token.group() == "{":
                depth += 1
            elif token.group() == "}":
                depth -= 1
            if depth == 0:
                text = text[:match.start()] + text[match.end():token.start()] + text[token.end():]
                count += 1
                break
        else:
            raise ValueError("Unclosed \\hl argument; no output has been written")
    return text, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path(__file__).with_name("Symbolic_LLM_HAR.tex"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.with_name(args.source.stem + "_sin_hl.tex")
    if output.resolve() == args.source.resolve():
        parser.error("The output must differ from the original")
    original = args.source.read_bytes()
    cleaned, count = remove_highlights(original.decode("utf-8"))
    with output.open("xb") as stream:
        stream.write(cleaned.encode("utf-8"))
    assert args.source.read_bytes() == original
    print(f"Removed {count} highlight wrappers. Copy: {output.resolve()}")


if __name__ == "__main__":
    main()
