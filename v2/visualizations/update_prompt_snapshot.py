"""Refresh the animation's embedded prompt viewer from the current agent/prompts.py. No model calls."""

import json
from pathlib import Path
import re
import runpy


def main():
    directory = Path(__file__).resolve().parent
    source = directory.parent / "src/agent/prompts.py"
    templates = runpy.run_path(str(source))["manifest"]()["prompts"]
    data = json.dumps(templates).replace("<", "\\u003c")
    page = directory / "instruction-flow.html"
    pattern = r'(<script id="prompt-data" type="application/json">).*?(</script>)'
    updated, count = re.subn(pattern, lambda m: m[1] + data + m[2], page.read_text(), flags=re.S)
    if count != 1:
        raise ValueError("expected exactly one embedded prompt snapshot")
    page.write_text(updated)
    (directory / "instruction-flow-preview.html").write_text(updated)
    print(f"Updated {len(templates)} prompt templates from {source}")


if __name__ == "__main__":
    main()
