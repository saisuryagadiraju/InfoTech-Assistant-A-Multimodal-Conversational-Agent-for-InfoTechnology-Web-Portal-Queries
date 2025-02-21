import json
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import openai
from flask import Flask, request, jsonify, render_template, session
import os
import whisper
import sounddevice as sd
import numpy as np
import wave
from flask import Flask, jsonify, request, send_file
import io

app = Flask(__name__)
app.secret_key = "AIT526" # Required for session management
    # A (RAG) chatbot for answering user queries by retrieving relevant content from scraped JSON file and  using an OpenAI local LLM for advanced responses.
# Initialize Whisper model
whisper_model = whisper.load_model("base")

class InteractiveRAGChatBot:
    def __init__(self, data_path: str):
        # Load JSON data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Initialize SpaCy model for keyword extraction and sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize SentenceTransformer for encoding and set device to CPU
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True)

        self.device = "cpu"
        self.embedding_model.to(self.device)

        # Set up local OpenAI-style LLM client with specified base URL and API key
        openai.api_base = "http://localhost:1234/v1"
        openai.api_key = "lm-studio"

        # Pre-define greetings
        self.greetings = ["hi", "hello", "hey", "Hii", "hii", "how are you", "how about you","quit", "exit", "bye"]
        self.greeting_response = {
            "default": "Hello! How can I assist you today?",
            "how_are_you": "I'm good, focusing on InfoTechnology!, Thank you",
            "how about you":"I'm Happy and learning new NDE Technologies",
            "farewell": "Thank you so much! Hope you got the necessary information."
        }
        # Process and prepare data chunks from JSON
        self.chunks = self.process_data()

    def handle_greetings(self, user_input: str) -> str:
        """
        Handle user greetings and respond accordingly.
        """
        user_input_lower = user_input.lower()
        if "how are you" in user_input.lower():
            return self.greeting_response["how_are_you"]
        if "how about you" in user_input.lower():
            return self.greeting_response["how about you"]
        if any(farewell in user_input_lower for farewell in ["quit", "exit", "bye"]):
            return self.greeting_response["farewell"]
        for greeting in self.greetings:
            if greeting in user_input.lower():
                return self.greeting_response["default"]
        return None


    def process_data(self) -> List[Dict]:
        # Store relevant information from JSON into manageable chunks
        chunks = []
        for id_key, content in self.data.items():
            if "text" in content:
                chunks.append({
                    "text": content["text"], #stores the text response
                    "id": id_key, #stores the post ids
                    "images": content.get("images", [])#stores the images
                })
        return chunks

    def extract_keywords(self, user_input: str) -> List[str]:
        # Extract keywords from user input
        doc = self.nlp(user_input)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        return keywords

    def search_data(self, keywords: List[str]) -> List[Dict]:
        # Find chunks containing any keyword to maximize relevant matches
        matched_chunks = []
        for chunk in self.chunks:
            if any(keyword.lower() in chunk["text"].lower() for keyword in keywords):
                matched_chunks.append(chunk)
        return matched_chunks

    def summarize_if_needed(self, text: str, max_length: int = 500) -> str:
        # Summarize content only if it exceeds a specified length
        if len(text) > max_length:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            return " ".join(sentences[:min(len(sentences), 6)])  # Limit to ~6 sentences
        return text

    def generate_response_from_chunks(self, user_input: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "I'm sorry, I couldn't find any relevant information for your query."

        # Select the best chunk based on keyword density and similarity
        query_embedding = self.embedding_model.encode([user_input], convert_to_numpy=True)
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)

        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        best_chunk = chunks[np.argmax(similarities)]

        # Provide full content if short, otherwise summarize
        detailed_text = self.summarize_if_needed(best_chunk["text"], max_length=700)

        # Include images if available
        images = best_chunk.get("images", [])[:3]# Can also adjust the images based on the response generation
        if images:
            images_html = "<br>".join([f'<img src="{img}" alt="Related Image" width="300">' for img in images])
            detailed_text += f"<br><br>{images_html}"

        return detailed_text

    def generate_response(self, user_input: str) -> str:
    
        #Generate a response to user input, including content from JSON data and a source link.
        # Base URL for generating dynamic source links
        base_url = "https://infotechnology.fhwa.dot.gov/"
        default_link = f"{base_url}bridge/"  # Fallback source link
        # If the user input matches any greeting pattern, return a predefined response.
        # Handle greetings
        greeting_response = self.handle_greetings(user_input)
        if greeting_response:
            return greeting_response

        # Extract keywords and search JSON data for relevant content
        keywords = self.extract_keywords(user_input)
        matched_chunks = self.search_data(keywords)
        chunk_response = ""
        source_link = default_link  # Default source link

        # Define a similarity threshold and use JSON data if relevant
        similarity_threshold = 0.5 # Minimum similarity score to consider a match relevant
        if matched_chunks:
            # Encode the user input and matched chunks for similarity comparison
            query_embedding = self.embedding_model.encode([user_input], convert_to_numpy=True)
            chunk_texts = [chunk["text"] for chunk in matched_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten() # Calculate similarity scores and find the best-matching chunk
            best_match_index = np.argmax(similarities)
            best_similarity_score = similarities[best_match_index]
            # If the best match is above the similarity threshold, prepare the chunk response
            if best_similarity_score >= similarity_threshold:
                best_chunk = matched_chunks[best_match_index]
                chunk_response = self.summarize_if_needed(best_chunk["text"], max_length=700)
                images = best_chunk.get("images", [])[:3]
                if images: # Append images if available in the chunk data
                    images_html = "<br>".join([f'<img src="{img}" alt="Related Image" width="300">' for img in images])
                    chunk_response += f"<br><br>{images_html}"

                # Generate source link dynamically based on keywords
                dynamic_path = "-".join(keywords).lower()
                source_link = f"{base_url}{dynamic_path}/"

        # Append source link to the chunk response
        if chunk_response:
            chunk_response += f"<br><br><strong>Source:</strong> <a href='{source_link}' target='_blank'>{source_link}</a>"

        # Generate a response from the LLM
        try: # Retrieve the conversation history from the session for context
            conversation_history = session.get('conversation_history', [])
            session_context = "\n".join(
                [f"User: {msg['content']}\nAssistant: {msg['content']}" for msg in conversation_history[-5:]]
            )# Combine session context and JSON context to form the prompt for the LLM
            prompt = f"{session_context}\n\nContext: {chunk_response}\nUser Question: {user_input}" if chunk_response else user_input
            # Call the LLM using the OpenAI-style API
            completion = openai.ChatCompletion.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7, # Adjust the randomness of the model's response
            )
            llm_response = completion.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error connecting to LLM: {e}")
            llm_response = "I'm having trouble connecting to my language model. Please try again later."

        # Combine JSON chunk response and LLM response
        combined_response = (
            f"{chunk_response}<br><br><strong>LLM Response:</strong><br>{llm_response}" 
            if chunk_response else llm_response
        )

        # Update the session with the new conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": combined_response})
        session['conversation_history'] = conversation_history  # Store updated history in session
        session.modified = True  # Ensure Flask saves the session data
        #Return the final combined response
        return combined_response


# Initialize the chatbot
chatbot = InteractiveRAGChatBot(data_path=r"C:\Users\sreed\OneDrive\Desktop\InfoTech Assistant\InfoTech Assistant\InfoBridge_scraped_data.json") 
# Need to specify the path correctly to load the files 

# Voice recording function


def record_audio(config=None):

    #Record audio with dynamic parameters for filename, duration, and sample rate.

    #Parameters:
        #config (dict): A dictionary with dynamic configuration options. Example: {"filename": "output.wav", "duration": 5, "fs": 16000}

    # Default configuration
    default_config = {
        "filename": "output.wav",
        "duration": 5,      # Default duration of 5 seconds
        "fs": 16000         # Default sample rate of 16 kHz (optimal for speech recognition)
    }

    # Update default configuration with user-provided values
    if config:
        default_config.update(config)

    filename = default_config["filename"]
    duration = default_config["duration"]
    fs = default_config["fs"]

    try:
        print(f"Recording... Duration: {duration} seconds, Sample Rate: {fs} Hz")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")

        # Save the recorded audio
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())

        print(f"Audio saved as '{filename}'")
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        raise

def transcribe_audio(filename='output.wav'):
    """
    Transcribe audio using Whisper and return the transcription text.

    Parameters:
        filename (str): Path to the audio file.

    Return:
        str: Transcription text.
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        print("Transcribing audio...")
        result = whisper_model.transcribe(filename)
        transcription = result.get("text", "").strip()

        if transcription:
            print("Transcription complete:", transcription)
            return transcription
        else:
            print("No speech detected in the audio.")
            return "No speech detected. Please try again."
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

# Flask route for voice-based chatbot interaction
@app.route('/voice_command', methods=['POST'])
def voice_command():
    try:
        # Dynamic configuration for recording
        config = {"filename": "voice_input.wav", "duration": 5, "fs": 16000}

        # Record audio
        record_audio(config=config)

        # Transcribe the recorded audio
        user_input = transcribe_audio(config["filename"])

        if not user_input or user_input.lower() == "no speech detected. please try again.":
            return jsonify({"error": "No speech detected. Please try again."})

        # Generate chatbot response based on transcription
        response = chatbot.generate_response(user_input)

        return jsonify({"user_input": user_input, "response": response})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})


@app.route('/')#Flask app which enroute to the HTML file for input and output
def index():
    return render_template("index.html")

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json.get("question", "")#to get the user input
    response = chatbot.generate_response(user_input)# post response to user
    return jsonify({"text": response})

@app.route('/scrape_data', methods=['POST'])
def scrape_data():
    return jsonify({"status": "Re-scraping initiated!"})#Rescraping initiation (Need best System Config that which can handle the process)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

