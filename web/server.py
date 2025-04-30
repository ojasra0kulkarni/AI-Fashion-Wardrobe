
from flask import Flask, request, Response, send_from_directory, jsonify
import subprocess
import os
import logging
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up logging (minimal)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Define paths
UPLOAD_FOLDER = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\Fashion Clothes"
FINAL_COMBINATIONS = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\Final_Combinations"
MAIN_PY = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\main.py"
AI_RESPONSE_FILE = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\ai_response.txt"

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Upload images
@app.route('/upload-images', methods=['POST'])
def upload_images():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Clear old images
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))
    for day in os.listdir(FINAL_COMBINATIONS):
        day_path = os.path.join(FINAL_COMBINATIONS, day)
        if os.path.isdir(day_path):
            for file in os.listdir(day_path):
                os.remove(os.path.join(day_path, file))
    
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    for file in request.files.getlist('images'):
        if os.path.splitext(file.filename)[1].lower() not in allowed_extensions:
            return {"error": "Invalid file type"}, 400
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return {"status": "ok"}, 200

# Run main.py and stream output
@app.route('/run-main', methods=['POST'])
def run_main():
    def generate():
        process = subprocess.Popen(['python', MAIN_PY], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output.strip():  # Yield only non-empty lines
                yield output
        process.stdout.close()
        return_code = process.poll()
        if return_code != 0:
            yield "Process failed\n"
    
    return Response(generate(), mimetype='text/plain')

# Get AI response
@app.route('/get-ai-response')
def get_ai_response():
    try:
        if not os.path.exists(AI_RESPONSE_FILE):
            return {"error": "No response found"}, 404
        with open(AI_RESPONSE_FILE, 'r', encoding='utf-8') as file:
            return {"text": file.read()}, 200
    except Exception as e:
        logger.error(f"AI response error: {str(e)}")
        return {"error": "Failed to read response"}, 500

# Get results
@app.route('/get-results')
def get_results():
    results = {}
    for day in [f"Day-{i}" for i in range(1, 8)]:
        day_path = os.path.join(FINAL_COMBINATIONS, day)
        if os.path.exists(day_path):
            images = [
                {"url": f"/images/{day}/{f}"}
                for f in os.listdir(day_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            results[day] = images
    return results

# Serve images
@app.route('/images/<day>/<filename>')
def serve_image(day, filename):
    return send_from_directory(os.path.join(FINAL_COMBINATIONS, day), filename)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Chat endpoint
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return {"reply": "No message provided"}, 400
        
        # Read AI response for context
        ai_response_text = ""
        if os.path.exists(AI_RESPONSE_FILE):
            with open(AI_RESPONSE_FILE, 'r', encoding='utf-8') as f:
                ai_response_text = f.read()
        
        prompt = f"Clothing data:\n{ai_response_text}\n\nQuestion: {user_message}"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 500,  # Reduced for shorter replies
                "temperature": 0.7
            }
        )
        return {"reply": response.text.strip()}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"reply": "Error occurred"}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
