#!/usr/bin/env python3
# Flask app for AI Companion using Gemini API
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
emotion_model = None
gemini_model = None

# Import personas
try:
    from presets import PERSONAS
except ImportError:
    logger.error("❌ Failed to load personas from 'presets' folder")
    raise

def initialize_gemini(persona_name="aria"):
    """Initialize Gemini API with selected persona"""
    global gemini_model
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("❌ GEMINI_API_KEY not found in environment variables")
        return False
    try:
        genai.configure(api_key=api_key)

        # Get system prompt from persona
        system_prompt = PERSONAS.get(persona_name, PERSONAS["aria"])

        gemini_model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 500,
            },
            system_instruction=system_prompt
        )
        logger.info(f"✅ Gemini model initialized successfully ({persona_name})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {e}")
        return False


def initialize_emotion_model():
    """Initialize emotion detection model"""
    global emotion_model
    try:
        emotion_model = pipeline(
            "text-classification", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1  # Use CPU
        )
        logger.info("✅ Emotion model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load emotion model: {e}")
        return False


# Initialize models when the app starts
def load_models():
    """Load all AI models"""
    logger.info("🤖 Loading AI models...")
    gemini_success = initialize_gemini()
    emotion_success = initialize_emotion_model()
    if not gemini_success:
        logger.warning("⚠️  Gemini API not available - check your API key")
    if not emotion_success:
        logger.warning("⚠️  Emotion model not available")
    logger.info("🚀 Model loading complete")


@app.route('/')
def home():
    return jsonify({
        "message": "AI Companion Backend is running!",
        "status": "success",
        "models": {
            "gemini_model": "loaded" if gemini_model else "failed",
            "emotion_model": "loaded" if emotion_model else "failed"
        },
        "version": "2.7.0-multi-persona-emotion"
    })


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        persona = data.get('persona', 'aria')  # Optional parameter

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        logger.info(f"User message: {user_message[:100]}...")

        # Re-initialize Gemini with selected persona
        if not gemini_model or not initialize_gemini(persona):
            return jsonify({
                "error": "Failed to initialize AI model with selected persona"
            }), 500

        # Analyze emotion
        emotion = None
        confidence = None
        if emotion_model:
            try:
                emotion_result = emotion_model(user_message)
                if emotion_result:
                    emotion = emotion_result[0]['label']
                    confidence = round(emotion_result[0]['score'], 2)
            except Exception as e:
                logger.error(f"Emotion analysis error: {e}")

        # Build emotion-aware message
        emotion_context = ""
        if emotion == "LABEL_2":
            emotion_context = "[The user seems happy]"
        elif emotion == "LABEL_1":
            emotion_context = "[The user seems neutral]"
        elif emotion == "LABEL_0":
            emotion_context = "[The user seems upset]"

        ai_response = generate_gemini_response(f"{user_message} {emotion_context}")

        if not ai_response:
            return jsonify({
                "error": "Gemini failed to generate response"
            }), 500

        response_data = {
            "response": ai_response,
            "emotion_detected": emotion,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "model_used": "gemini",
            "persona": persona
        }

        logger.info(f"AI response generated successfully (persona: {persona}, emotion: {emotion})")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


def generate_gemini_response(message):
    """Generate response using Gemini API"""
    if not gemini_model:
        logger.warning("Gemini model not available")
        return None
    try:
        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(message)
        if response and response.text:
            logger.info("✅ Gemini response generated")
            return response.text.strip()
        else:
            logger.warning("Empty response from Gemini")
            return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


@app.route('/emotion', methods=['POST'])
def analyze_emotion():
    """Separate endpoint for emotion analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not emotion_model:
            return jsonify({"error": "Emotion model not available"}), 503
        result = emotion_model(text)
        return jsonify({
            "emotion": result[0]['label'],
            "confidence": round(result[0]['score'], 2),
            "all_scores": result,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "gemini_model": "available" if gemini_model else "unavailable",
            "emotion_model": "available" if emotion_model else "unavailable"
        },
        "environment": {
            "gemini_api_key_configured": bool(os.getenv('GEMINI_API_KEY')),
            "python_version": os.sys.version
        }
    })


@app.route('/test-gemini', methods=['POST'])
def test_gemini():
    """Test endpoint for Gemini API"""
    try:
        data = request.get_json()
        test_message = data.get('message', 'Hello, are you working?')
        if not gemini_model:
            return jsonify({
                "status": "error",
                "message": "Gemini model not initialized",
                "api_key_configured": bool(os.getenv('GEMINI_API_KEY'))
            }), 503
        response = generate_gemini_response(test_message)
        return jsonify({
            "status": "success" if response else "error",
            "test_message": test_message,
            "gemini_response": response,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# Initialize models when the app starts
load_models()


if __name__ == '__main__':
    from pyngrok import ngrok, conf

    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")

    if NGROK_AUTH_TOKEN and NGROK_DOMAIN:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        try:
            public_url = ngrok.connect(
                addr=5000,
                proto="http",
                domain=NGROK_DOMAIN
            )
            print(f"🌐 ngrok tunnel active at: {public_url}")
        except Exception as e:
            print(f"❌ Failed to start ngrok tunnel: {e}")
    else:
        print("⚠️  NGROK_AUTH_TOKEN or NGROK_DOMAIN not set. Skipping ngrok tunnel.")

    print("🚀 Starting AI Companion Backend with Gemini API...")
    print("📍 Backend running at: http://localhost:5000")
    print("🤖 Gemini API Status:", "✅ Ready" if gemini_model else "❌ Not configured")
    print("🎭 Emotion Model Status:", "✅ Ready" if emotion_model else "❌ Not available")
    print("💡 Make sure to set your GEMINI_API_KEY in your .env file!")
    print("💡 Get your API key from: https://makersuite.google.com/app/apikey") 
    app.run(debug=True, host='0.0.0.0', port=5000)