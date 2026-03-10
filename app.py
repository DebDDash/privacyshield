"""
app.py — Flask web UI for PrivacyShield
Upload a PDF → process it → download the result.
(Currently returns the same PDF; swap in real pipeline later.)
"""

import os
import uuid
import shutil
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    after_this_request,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page upload UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a PDF, process it, return download metadata."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # Save upload with a unique id so filenames never collide
    job_id = uuid.uuid4().hex
    original_name = file.filename
    safe_stem = Path(original_name).stem
    upload_path = UPLOAD_DIR / f"{job_id}.pdf"
    file.save(str(upload_path))

    # ── processing stub ──────────────────────────────────────────────────
    # TODO: plug real pipeline here:
    #   from privacyshield.pipeline import run_pipeline
    #   result = run_pipeline(str(upload_path))
    #
    # For now we simply copy the input PDF as the "processed" output.
    output_path = OUTPUT_DIR / f"{job_id}_redacted.pdf"
    shutil.copy2(str(upload_path), str(output_path))
    # ─────────────────────────────────────────────────────────────────────

    return jsonify({
        "job_id": job_id,
        "original_name": original_name,
        "download_name": f"{safe_stem}_redacted.pdf",
    })


@app.route("/download/<job_id>")
def download(job_id):
    """Stream the processed PDF back to the browser."""
    # Basic input validation
    if not job_id.isalnum():
        return jsonify({"error": "Invalid job id."}), 400

    output_path = OUTPUT_DIR / f"{job_id}_redacted.pdf"
    upload_path = UPLOAD_DIR / f"{job_id}.pdf"

    if not output_path.exists():
        return jsonify({"error": "File not found. It may have expired."}), 404

    download_name = request.args.get("name", "redacted.pdf")

    @after_this_request
    def cleanup(response):
        """Remove temp files after sending."""
        try:
            output_path.unlink(missing_ok=True)
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass
        return response

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=download_name,
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
