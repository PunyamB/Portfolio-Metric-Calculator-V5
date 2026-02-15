import os
import subprocess
import json


def generate_report(report_data, narrative, output_path="portfolio_report.docx"):
    """Generate report DOCX from data and narrative."""
    combined = {
        "data": report_data,
        "narrative": narrative,
    }

    json_path = output_path.replace(".docx", ".json")
    with open(json_path, "w") as f:
        json.dump(combined, f, default=str)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "report_builder.js")

    try:
        result = subprocess.run(
            ["node", script_path, json_path, output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise Exception(f"Node error: {result.stderr}")
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)

    return output_path
