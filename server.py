"""
================================================================================
TB DETECTION SYSTEM — server_optimized.py
================================================================================
Optimized Flask API server with GPU acceleration and enhanced features.
Works with both model.py and model_optimized.py

FEATURES:
- ✅ GPU-accelerated predictions
- ✅ Batch prediction support
- ✅ CORS enabled for web apps
- ✅ Detailed logging
- ✅ Image preprocessing validation
- ✅ Multiple image format support
- ✅ Health check endpoint
- ✅ Model info endpoint

USAGE:
    # 1. Make sure you have a trained model
    python model_optimized.py  # or python model.py

    # 2. Start the server
    python server_optimized.py                # default port 5000
    python server_optimized.py --port 8080    # custom port

ENDPOINTS:
    GET  /health       - Health check
    GET  /model-info   - Model metadata
    POST /predict      - Single image prediction
    POST /predict-batch - Multiple images prediction

EXAMPLE REQUESTS:
    # Health check
    curl http://localhost:5000/health

    # Single prediction
    curl -X POST -F "image=@xray.png" http://localhost:5000/predict

    # Batch prediction
    curl -X POST -F "images=@xray1.png" -F "images=@xray2.png" \
         http://localhost:5000/predict-batch
================================================================================
"""

import os
import argparse
import numpy as np
import cv2
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Try to import from optimized model first, fall back to original
try:
    from model_final_fast import TBPredictor, Config
    print("✓ Using optimized model (model_final_fast.py)")
except ImportError:
    try:
        from model import TBPredictor, Config
        print("✓ Using original model (model.py)")
    except ImportError:
        print("ERROR: Could not import TBPredictor from model.py or model_optimized.py")
        exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = 10

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

predictor = TBPredictor()
MODEL_LOADED = False
MODEL_METADATA = None

try:
    logger.info("Loading TB detection model...")
    predictor.load_model()
    MODEL_LOADED = True
    MODEL_METADATA = predictor.metadata
    logger.info("✓ Model loaded successfully")
    logger.info(f"  Model timestamp: {MODEL_METADATA.get('timestamp', 'Unknown')}")
    logger.info(f"  Test accuracy: {MODEL_METADATA.get('training_metrics', {}).get('test_accuracy', 0)*100:.2f}%")
except FileNotFoundError as e:
    logger.error(f"Model not found: {e}")
    logger.error("Run 'python model_optimized.py' first to train and save a model.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_size(file):
    """Check if file size is within limits"""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    return size <= MAX_IMAGE_SIZE


def decode_image(file):
    """
    Decode uploaded file into BGR numpy array
    Returns: (img_bgr, error_message)
    """
    try:
        # Read file bytes
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        
        # Decode to BGR image
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return None, "Could not decode image. File may be corrupted."
        
        # Validate image dimensions
        h, w = img_bgr.shape[:2]
        if h < 50 or w < 50:
            return None, f"Image too small ({w}x{h}). Minimum size: 50x50 pixels."
        
        if h > 5000 or w > 5000:
            return None, f"Image too large ({w}x{h}). Maximum size: 5000x5000 pixels."
        
        return img_bgr, None
        
    except Exception as e:
        return None, f"Image decode failed: {str(e)}"


def format_prediction_response(result, processing_time=None):
    """Format prediction result into consistent JSON response"""
    response = {
        "success": True,
        "prediction": result['prediction'],
        "confidence": round(result['confidence'], 4),
        "probabilities": {
            k: round(v, 4) for k, v in result['probabilities'].items()
        },
        "risk_level": result['risk_level'],
        "timestamp": datetime.now().isoformat()
    }
    
    if processing_time:
        response['processing_time_ms'] = round(processing_time * 1000, 2)
    
    return response


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """API information endpoint"""
    return jsonify({
        "service": "TB Detection API",
        "version": "2.0 (Optimized)",
        "model_loaded": MODEL_LOADED,
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model-info",
            "predict": "POST /predict",
            "predict_batch": "POST /predict-batch"
        },
        "documentation": "Send POST to /predict with multipart/form-data containing 'image' field"
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "gpu_available": Config.USE_CUDA if MODEL_LOADED else False
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return detailed model information"""
    if not MODEL_LOADED or not MODEL_METADATA:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    metrics = MODEL_METADATA.get('training_metrics', {})
    config = MODEL_METADATA.get('config', {})
    
    return jsonify({
        "model_name": MODEL_METADATA.get('model_name', 'Unknown'),
        "timestamp": MODEL_METADATA.get('timestamp', 'Unknown'),
        "performance": {
            "test_accuracy": round(metrics.get('test_accuracy', 0) * 100, 2),
            "sensitivity": round(metrics.get('sensitivity', 0) * 100, 2),
            "specificity": round(metrics.get('specificity', 0) * 100, 2),
            "precision": round(metrics.get('precision', 0) * 100, 2),
            "f1_score": round(metrics.get('f1_score', 0) * 100, 2),
        },
        "configuration": {
            "categories": config.get('categories', []),
            "image_size": config.get('img_size', 224),
            "gpu_used": config.get('gpu_used', False),
            "gpu_name": config.get('gpu_name', 'N/A'),
            "num_workers": config.get('num_workers', 1)
        },
        "training_time": metrics.get('training_time', {})
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single image prediction endpoint
    
    Expected: multipart/form-data with 'image' field
    Returns: JSON with prediction results
    """
    start_time = time.time()
    
    # ── Check if model is loaded ──────────────────────────────────────────
    if not MODEL_LOADED:
        logger.error("Prediction request received but model not loaded")
        return jsonify({
            "success": False,
            "error": "Model not loaded. Train first with: python model_optimized.py"
        }), 503
    
    # ── Validate request ──────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No 'image' field in request",
            "hint": "Send multipart/form-data with field name 'image'"
        }), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400
    
    # ── Validate file type ────────────────────────────────────────────────
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # ── Validate file size ────────────────────────────────────────────────
    if not validate_image_size(file):
        return jsonify({
            "success": False,
            "error": f"File too large. Maximum size: {MAX_IMAGE_SIZE / (1024*1024):.1f}MB"
        }), 400
    
    # ── Decode image ──────────────────────────────────────────────────────
    img_bgr, error = decode_image(file)
    if error:
        logger.warning(f"Image decode failed: {error}")
        return jsonify({
            "success": False,
            "error": error
        }), 400
    
    # ── Run prediction ────────────────────────────────────────────────────
    try:
        logger.info(f"Processing image: {secure_filename(file.filename)}")
        result = predictor.predict(img_bgr, return_details=True)
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction: {result['prediction']} "
                   f"(confidence: {result['confidence']*100:.1f}%) "
                   f"in {processing_time*1000:.1f}ms")
        
        return jsonify(format_prediction_response(result, processing_time)), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected: multipart/form-data with multiple 'images' fields
    Returns: JSON array with prediction results for each image
    """
    start_time = time.time()
    
    # ── Check if model is loaded ──────────────────────────────────────────
    if not MODEL_LOADED:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 503
    
    # ── Get all uploaded files ────────────────────────────────────────────
    if "images" not in request.files:
        return jsonify({
            "success": False,
            "error": "No 'images' field in request",
            "hint": "Send multipart/form-data with one or more 'images' fields"
        }), 400
    
    files = request.files.getlist("images")
    
    if len(files) == 0:
        return jsonify({
            "success": False,
            "error": "No images provided"
        }), 400
    
    if len(files) > MAX_BATCH_SIZE:
        return jsonify({
            "success": False,
            "error": f"Too many images. Maximum batch size: {MAX_BATCH_SIZE}"
        }), 400
    
    # ── Process each image ────────────────────────────────────────────────
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            # Validate file
            if not allowed_file(file.filename):
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            if not validate_image_size(file):
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": "File too large"
                })
                continue
            
            # Decode image
            img_bgr, error = decode_image(file)
            if error:
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": error
                })
                continue
            
            # Predict
            result = predictor.predict(img_bgr, return_details=True)
            results.append({
                "index": idx,
                "filename": secure_filename(file.filename),
                **format_prediction_response(result)
            })
            
        except Exception as e:
            errors.append({
                "index": idx,
                "filename": file.filename,
                "error": str(e)
            })
    
    processing_time = time.time() - start_time
    
    logger.info(f"Batch prediction: {len(results)} successful, "
               f"{len(errors)} failed in {processing_time*1000:.1f}ms")
    
    return jsonify({
        "success": True,
        "total_images": len(files),
        "successful": len(results),
        "failed": len(errors),
        "processing_time_ms": round(processing_time * 1000, 2),
        "results": results,
        "errors": errors if errors else None
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size: {MAX_IMAGE_SIZE / (1024*1024):.1f}MB"
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "success": False,
        "error": "Internal server error occurred"
    }), 500


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TB Detection API Server (Optimized)")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port to listen on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  TB DETECTION API SERVER (OPTIMIZED)")
    print(f"{'='*80}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Model loaded: {'✓ YES' if MODEL_LOADED else '✗ NO'}")
    if MODEL_LOADED:
        print(f"  GPU acceleration: {'✓ ENABLED' if Config.USE_CUDA else '✗ DISABLED'}")
        if MODEL_METADATA:
            metrics = MODEL_METADATA.get('training_metrics', {})
            print(f"  Test accuracy: {metrics.get('test_accuracy', 0)*100:.2f}%")
    print(f"{'='*80}")
    print(f"\n  API Endpoints:")
    print(f"  • http://{args.host}:{args.port}/          - API info")
    print(f"  • http://{args.host}:{args.port}/health    - Health check")
    print(f"  • http://{args.host}:{args.port}/model-info - Model details")
    print(f"  • http://{args.host}:{args.port}/predict   - Single prediction")
    print(f"  • http://{args.host}:{args.port}/predict-batch - Batch prediction")
    print(f"\n  Example:")
    print(f"  curl -X POST -F 'image=@xray.png' http://localhost:{args.port}/predict")
    print(f"\n{'='*80}\n")

    # Set max upload size
    app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE
    port=init(os.environ.get("PORT",args.port))
    # Run server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True  # Enable multi-threading for concurrent requests
    )
