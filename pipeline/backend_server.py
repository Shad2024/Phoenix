from flask import Flask, request, jsonify , send_from_directory
from flask import send_file 
from flask_cors import CORS
import json
import threading
import time
import pandas as pd
import os
from analyze_images import run_damage_analysis 
import subprocess
from VoiceAssistant import speak, start_listening, stop

CURRENT_ANALYSIS_FILE = os.path.join(os.path.dirname(__file__), "current_analysis.json")
CAPTURED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'captured_images') 
BASE_CAPTURED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "captured_images")


def run_backend_command(cmd):
    """
    Execute the Python logic based on the command and return the next UI action.
    """
    cmd = cmd.lower()
    print(f"[DEBUG] Executing command: {cmd}")

    base_dir = os.path.dirname(__file__)
    python_exe = os.path.join(base_dir, "..", ".venv", "Scripts", "python.exe")
    python_exe = os.path.normpath(python_exe)

    # Damage analysis module - The key change
    if ("التحليل" in cmd and "الأضرار" in cmd) or ("damage analysis" in cmd) or ("start analysis" in cmd and "damage" in cmd):
        speak("تشغيل وحدة تحليل الأضرار.")
        target = os.path.join(base_dir, "analyze_images.py")
        subprocess.Popen([python_exe, target], cwd=base_dir)
        # Return the specific action that the Flutter app will read
        return "GO_TO_DAMAGE_ANALYSIS" 

    # Capture images module
    elif ("ابدأ" in cmd and "التصوير" in cmd) or ("start capturing" in cmd and "camera" in cmd):
        speak("تشغيل وحدة التقاط الصور.")
        target = os.path.join(base_dir, "capture_images.py")
        subprocess.Popen([python_exe, target], cwd=base_dir)
        return "STAY_ON_INPUT" 
    
    # Material analysis module
    elif ("ابدأ التحليل" in cmd and "المواد" in cmd) or("material analysis" in cmd) or("start analysis" in cmd and "materials" in cmd):
        speak("تشغيل وحدة تحليل المواد.")
        target = os.path.join(base_dir, "Waste_Material.py") 
    if os.path.exists(target):
        subprocess.Popen([python_exe, target], cwd=base_dir) 
        return "GO_TO_MATERIAL_ANALYSIS"

    # Stop everything
    elif "توقف" in cmd or "stop" in cmd: 
        # ... (existing logic) ...
        return "EXIT_APP" 

    # If no specific command is matched
    return "STAY_ON_INPUT"

def handle_voice_command(cmd):

    run_backend_command(cmd)

# ---------------------- FLASK BACKEND ----------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

listener_thread = None
is_listening = False
lock = threading.Lock()


def start_voice_listener():
    """Start the Vosk voice listener in background thread."""
    global is_listening
    with lock:
        if is_listening:
            return
        is_listening = True

    speak("تم تفعيل فنيكس.")
    print("🎤 Phoenix voice listener started")

    start_listening(handle_voice_command)


# --- New Endpoint to Get All Analysis Data ---
@app.route('/analysis_info', methods=['GET'])
def get_analysis_info():
    if not os.path.exists(CURRENT_ANALYSIS_FILE):
        return jsonify({"error": "Current analysis data not found"}), 404
        
    try:
        with open(CURRENT_ANALYSIS_FILE, 'r') as f:
            full_analysis_data = json.load(f)
            
            # Extract the list of filenames in the correct sequence
            image_list = [item['filename'] for item in full_analysis_data]
            
            return jsonify({
                "image_list": image_list,
                "analysis_details": full_analysis_data # Send the full data structure
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Failed to load analysis data: {str(e)}"}), 500

# --- Updated Image Serving Endpoint ---
@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename): 
    # 1. Check the BASE directory for the latest timestamped folder
    if not os.path.exists(BASE_CAPTURED_DIR) or not os.listdir(BASE_CAPTURED_DIR):
        print(f"ERROR: Base captured images directory not found: {BASE_CAPTURED_DIR}")
        return jsonify({"error": "Base captured images directory not found"}), 404

    try:
        # Find the latest subfolder by modification time
        latest_folder_path = max(
            [os.path.join(BASE_CAPTURED_DIR, d) for d in os.listdir(BASE_CAPTURED_DIR)],
            key=os.path.getmtime
        )
    except Exception as e:
        print(f"ERROR: Could not determine latest capture folder: {e}")
        return jsonify({"error": "Could not determine latest capture folder"}), 404

    # 2. Construct the full path using the LATEST FOLDER and the filename from the URL
    image_path = os.path.join(latest_folder_path, filename)
    
    print(f"DEBUG: Attempting to serve image from: {image_path}") # <-- CHECK THIS PATH!

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image {filename} not found at expected path: {image_path}"}), 404

    # Flask sends the file bytes directly
    return send_file(image_path, mimetype='image/png')

# Use this new, unified endpoint
@app.route("/command", methods=["POST"])
def handle_command():
    """Handle all commands (text, or initial voice activation) from frontend."""
    data = request.get_json()
    command_type = data.get("command_type", "text") 
    cmd = data.get("command", "").strip()

    if not cmd:
        return jsonify({"error": "Empty command"}), 400

    print(f"[{command_type.upper()}] Command received: {cmd}")
    
    # Check for voice activation first
    if command_type == 'voice:activate':
        # ... (your existing voice activation logic) ...
        # (Assuming start_voice_listener does not block)
        start_voice_listener()
        return jsonify({
            "status": "processed",
            "command": cmd,
            "next_ui_action": "VOICE_LISTENING"
        })

    # --- CORE LOGIC CHANGE FOR DAMAGE ANALYSIS ---
    
    # 1. Determine the intended next action by calling the existing logic
    next_action = run_backend_command(cmd)

    if next_action == "GO_TO_DAMAGE_ANALYSIS":
        print("[BLOCKING] Waiting for damage analysis to write first result...")
        
        # NOTE: Your analyze_images.py must be modified to DELETE the 
        # CURRENT_ANALYSIS_FILE at the start of its run to make this wait reliable!
        
        timeout = 30 # Max time to wait for the first image result
        start_time = time.time()
        
        # Block and wait for the file to be created by the analyze_images.py process
        while not os.path.exists(CURRENT_ANALYSIS_FILE) and (time.time() - start_time) < timeout:
            time.sleep(0.5) # Check every half second
        
        if os.path.exists(CURRENT_ANALYSIS_FILE):
            # Success: File exists, analysis started, OK to switch view
            return jsonify({
                "status": "processed",
                "command": cmd,
                "next_ui_action": next_action # GO_TO_DAMAGE_ANALYSIS
            }), 200
        else:
            # Failure: Timeout reached or model crashed before writing file
            print("❌ Analysis process timed out or failed to produce initial file.")
            # Return an error action or a 500 status
            return jsonify({
                "status": "error",
                "message": "Analysis failed to start or timed out.",
                "next_ui_action": "DISPLAY_ERROR"
            }), 500

    # --- HANDLE MATERIAL ANALYSIS BLOCKING ---
    if next_action == "GO_TO_MATERIAL_ANALYSIS":
        # Define your CSV path
        SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "segmentation_summary.csv")
        
        # Clear old results to ensure we wait for FRESH data
        if os.path.exists(SUMMARY_CSV):
            os.remove(SUMMARY_CSV)

        print("[BLOCKING] Waiting for YOLO Material Segmentation to start...")
        
        timeout = 40  # YOLO might take longer to load than damage model
        start_time = time.time()
        
        # Wait for the script to create the file
        while not os.path.exists(SUMMARY_CSV) and (time.time() - start_time) < timeout:
            time.sleep(1.0) 

        if os.path.exists(SUMMARY_CSV):
            return jsonify({
                "status": "processed",
                "next_ui_action": "GO_TO_MATERIAL_ANALYSIS"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Material analysis timed out.",
                "next_ui_action": "DISPLAY_ERROR"
            }), 500

    return jsonify({
        "status": "processed",
        "command": cmd,
        "next_ui_action": next_action 
    })
    

@app.route("/stop", methods=["POST"])
def stop_all():
    """Stop all backend processes."""
    global is_listening
    is_listening = False
    stop()
    print("🛑 Phoenix stopped")
    return jsonify({"status": "stopped"})

@app.route("/analysis/status", methods=["GET"])
def get_analysis_status():
    """
    Reads the current analysis results from the file generated by analyze_images.py.
    """
    if not os.path.exists(CURRENT_ANALYSIS_FILE):
        return jsonify({"error": "Analysis not running or no data yet"}), 404

    try:
        with open(CURRENT_ANALYSIS_FILE, 'r') as f:
            data = json.load(f)
        
        # Add the URL for the image
        data["analysis_image_url"] = f"/analysis/image/{data['filename']}"
        
        return jsonify({
            "status": "success",
            "data": data
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Error reading analysis data file"}), 500


@app.route('/analysis/image/<filename>', methods=['GET'])
def serve_analysis_image(filename):
    """
    Serves the specific image file being referenced in the analysis.
    This assumes analyze_images.py saves the images in a predictable structure.
    """
    # This is a critical assumption: Re-construct the latest captured image path
    # based on the filename provided by the status endpoint.
    BASE_CAPTURED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "captured_images")
    
    # 1. Find the latest folder (same logic as in your original main())
    if not os.path.exists(BASE_CAPTURED_DIR) or not os.listdir(BASE_CAPTURED_DIR):
        return jsonify({"error": "Captured images directory not found"}), 404
    
    try:
        # Find the latest subfolder by modification time
        latest_folder_path = max(
            [os.path.join(BASE_CAPTURED_DIR, d) for d in os.listdir(BASE_CAPTURED_DIR)],
            key=os.path.getmtime
        )
    except:
        return jsonify({"error": "Could not determine latest capture folder"}), 404

    # 2. Construct the full path
    image_path = os.path.join(latest_folder_path, filename)

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file not found: {filename}"}), 404
    
    # 3. Serve the file
    return send_file(image_path, mimetype='image/jpeg')


SUMMARY_CSV = r"C:\Users\bolaky\PycharmProjects\Rebuild\outputs\segmentation_summary.csv"
MATERIAL_RESULTS_DIR = r"C:\Users\bolaky\PycharmProjects\Rebuild\outputs\Material_results"

@app.route('/material_analysis_data', methods=['GET'])
def get_material_data():
    if not os.path.exists(SUMMARY_CSV):
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(SUMMARY_CSV)
    
    results = []
    # Group by image_name to keep materials of the same image together
    for img_name, group in df.groupby("image_name"):

        base_name = os.path.splitext(img_name)[0]
        detected_filename = f"{base_name}_detected.jpg"
        
        results.append({
            "image_path": detected_filename,
            "materials": group["material"].tolist(),
            "percentages": group["percentage"].tolist()
        })
    
    # This sends a list of images, each containing its own list of materials
    return jsonify(results)

@app.route('/material_image/<filename>')
def serve_material_image(filename):
    return send_from_directory(MATERIAL_RESULTS_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "listening": is_listening
    })


if __name__ == "__main__":
    print("🚀 Phoenix backend running (Flask)")
    app.run(host="0.0.0.0", port=5000, debug=True)