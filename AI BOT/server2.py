from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai

app = Flask(__name__)

# Load your clothing data from ai_response.txt with absolute path
with open(r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\ai_response.txt", 'r', encoding='utf-8') as f:
    ai_response_text = f.read()

# Setup Gemini API with your API key
genai.configure(api_key="AIzaSyAnqed_hXXm60m2pwav0V1cFi43Dnc1sWw")  # Using the same key from AI_API_Suggestions.py

@app.route('/')
def index():
    return send_from_directory('.', 'index2.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message')
    prompt = f"Here is the clothing data:\n{ai_response_text}\n\nAnswer this question: {user_message}"

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 2000,
                "temperature": 0.7
            }
        )
        bot_reply = response.text.strip()
        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(e)
        return jsonify({'reply': 'Sorry, something went wrong. Please try again later.'})

if __name__ == "__main__":
    app.run(debug=True)
