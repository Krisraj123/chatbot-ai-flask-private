import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ai  # Import the ai module
import os
# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)


UPLOAD_FOLDER = '/home/yourusername/uploads'
app.config['UPLOAD_FOLDER']

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    try:
        user_message = request.json.get('userMessage')  # Extract the user's message from the request
        if not user_message:
            return jsonify({"error": "User message is required."}), 400

        bot_response = ai.process_prompt(user_message)  # Process the user's message using the ai module

        # Return the bot's response as JSON
        return jsonify({"botResponse": bot_response}), 200
    except Exception as e:
        app.logger.error(f"Error processing message: {e}")
        return jsonify({"error": "An error occurred while processing the message."}), 500

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "botResponse": "It seems like the file was not uploaded correctly. Can you try again? If the problem persists, try using a different file."
            }), 400

        file = request.files['file']  # Extract the uploaded file from the request
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) # Define the path where the file will be saved
        file.save(file_path)  # Save the file

        ai.process_document(file_path)  # Process the document using the ai module

        # Return a success message as JSON
        return jsonify({
            "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any questions regarding it!"
        }), 200
    except Exception as e:
        app.logger.error(f"Error processing document: {e}")
        return jsonify({"error": "An error occurred while processing the document."}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)
