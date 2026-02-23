from flask import Flask, request, jsonify
import subprocess
import os
from pathlib import Path
import json

app = Flask(__name__)

# Determine project paths
BASE_DIR = Path(__file__).parent.parent  # Assuming api.py is in pipeline/
SEGMENTATION_CSV = BASE_DIR / "segmentation_results.csv"
CAPTURED_DIR = BASE_DIR / "outputs" / "captured_images"
PIPELINE_DIR = BASE_DIR / "pipeline"


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Phoenix API is running", "endpoints": ["/data/images", "/analyze/material", "/analyze/damage", "/voice/command"]})


@app.route('/analyze/material', methods=['POST'])
def analyze_material():
    """Trigger material analysis pipeline."""
    try:
        # Run Waste_Material.py
        result = subprocess.run(
            ["python", str(PIPELINE_DIR / "Waste_Material.py")],
            cwd=str(PIPELINE_DIR),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Load and return results from segmentation_results.csv
            if SEGMENTATION_CSV.exists():
                # Simple parsing (adjust as needed)
                with open(SEGMENTATION_CSV, 'r') as f:
                    data = f.read()
                return jsonify({"status": "success", "data": data})
            else:
                return jsonify({"status": "success", "message": "Analysis complete, check CSV"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/analyze/damage', methods=['POST'])
def analyze_damage():
    """Trigger damage analysis pipeline."""
    try:
        # Run analyze_images.py
        result = subprocess.run(
            ["python", str(PIPELINE_DIR / "analyze_images.py")],
            cwd=str(PIPELINE_DIR),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return jsonify({"status": "success", "message": "Damage analysis complete"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/data/images', methods=['GET'])
def get_images():
    """Get list of latest captured images."""
    try:
        if CAPTURED_DIR.exists():
            subfolders = [f for f in CAPTURED_DIR.iterdir() if f.is_dir()]
            if subfolders:
                latest_folder = max(subfolders, key=os.path.getmtime)
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend([str(img)
                                  for img in latest_folder.glob(ext)])
                return jsonify({"images": images})
        return jsonify({"images": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/voice/command', methods=['POST'])
def voice_command():
    """Handle voice commands."""
    data = request.json
    cmd = data.get('command', '').lower()

    python_exe = str(BASE_DIR / ".venv" / "Scripts" / "python.exe")

    # Capture images module
    if ("ابدأ" in cmd and "التصوير" in cmd) or ("start capturing" in cmd and "camera" in cmd):
        message = "تشغيل وحدة التقاط الصور."
        target = str(PIPELINE_DIR / "capture_images.py")
        subprocess.Popen([python_exe, target], cwd=str(BASE_DIR))
        return jsonify({"status": "success", "message": message, "action": "start_capturing"})

    # Damage analysis module
    elif ("التحليل" in cmd and "الأضرار" in cmd) or ("start analysis" in cmd and "damage" in cmd):
        message = "تشغيل وحدة تحليل الأضرار."
        target = str(PIPELINE_DIR / "analyze_images.py")
        subprocess.Popen([python_exe, target], cwd=str(BASE_DIR))
        return jsonify({"status": "success", "message": message, "action": "start_damage_analysis"})

    # Material analysis module
    elif ("ابدأ التحليل" in cmd and "المواد" in cmd) or ("start analysis" in cmd and "materials" in cmd):
        message = "تشغيل وحدة تحليل المواد."
        target = str(PIPELINE_DIR / "Waste_Material.py")
        subprocess.Popen([python_exe, target], cwd=str(BASE_DIR))

        # After it finishes, run your recommendation model
        target_recommendation = str(
            PIPELINE_DIR / "Ara_recommend_materials.py")
        subprocess.Popen([python_exe, target_recommendation],
                         cwd=str(BASE_DIR))

    else:
        return jsonify({"status": "unknown command"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
