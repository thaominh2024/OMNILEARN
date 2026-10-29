"""Minimal coordinator: PDF -> units -> optional story API calls.

Runs the original CLI pipeline unchanged, then calls the separately-running
story service. The story endpoint is TOPIC-BASED, not grounded retrieval.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    ap.add_argument("--story-url", default="http://127.0.0.1:8006/story")
    ap.add_argument("--generate-stories", action="store_true")
    ap.add_argument("--unit-limit", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).parent / "src" / "pipeline.py"),
           "--pdf", str(args.pdf), "--out_dir", str(args.out_dir),
           "--run_id", "portfolio_demo", "--unit_limit", str(args.unit_limit)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(data["error"])
    (args.out_dir / "units.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated units:", json.dumps(data["units"], ensure_ascii=False, indent=2))
    if args.generate_stories:
        stories = []
        for unit in data["units"]:
            payload = json.dumps({"unit": unit}).encode("utf-8")
            req = urllib.request.Request(args.story_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as response:
                story = json.load(response)["text"]
            stories.append({"unit": unit, "story": story})
        path = args.out_dir / "stories.json"
        path.write_text(json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", path)


if __name__ == "__main__":
    main()
