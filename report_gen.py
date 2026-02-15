import os
import subprocess
import json
import tempfile


def _generate_docx_via_node(report_data, narrative, output_path, report_type="weekly"):
    """Generate DOCX by writing data to JSON and calling Node.js script."""
    # combine data + narrative into single JSON
    combined = {
        "report_type": report_type,
        "data": report_data,
        "narrative": narrative,
    }

    # write to temp JSON
    json_path = output_path.replace(".docx", ".json")
    with open(json_path, "w") as f:
        json.dump(combined, f, default=str)

    # find the node script
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


def generate_weekly_report(report_data, narrative, output_path="weekly_report.docx"):
    """Generate weekly report DOCX."""
    return _generate_docx_via_node(report_data, narrative, output_path, "weekly")


def generate_inception_report(report_data, narrative, output_path="inception_report.docx"):
    """Generate inception report DOCX."""
    return _generate_docx_via_node(report_data, narrative, output_path, "inception")
