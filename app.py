from flask import Flask, render_template,request,jsonify
import requests
import logging
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources ={r"/*" : {"origins" : '*'}})
app.logger.setLevel(logging.ERROR)

colab_doc_api_url = "https://9ea3-35-229-64-255.ngrok-free.app/api/aidoc"
colab_prompt_api_url = "https://9ea3-35-229-64-255.ngrok-free.app/api/aiprompt"

@app.route('/' , methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/process-message' , methods=['POSt'])
def process_message_route():
    user_message = request.json['userMessage']
    print('user_message' , user_message)
    try:
        response = requests.post(colab_prompt_api_url, json={'userMessage': user_message})
        response.raise_for_status()
        ai_response = response.json()['botResponse']
        return jsonify({"botResponse": ai_response}), 200

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with AI: {e}")
        return jsonify({"botResponse": f"Error communicating with AI"}), 500



@app.route('/process-document' , methods = ['POST'])
def process_document_route():
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "No file provided"
        }),400

    file = request.files['file']
    try:
        response = requests.post(colab_doc_api_url, files = {'file':file})
        response.raise_for_status()
        ai_response = response.json()['botResponse']
        return jsonify({
            "botResponse": ai_response

        }), 200
    except:
        return jsonify({"botResponse": f"Error communicating with AI: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
