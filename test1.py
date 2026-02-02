"""
================================================================================
TB DETECTION SYSTEM — test_client.py
================================================================================
Test client for the TB Detection API server.
Demonstrates how to make POST requests to the server.

USAGE:
    # 1. Start the server first
    python server_optimized.py

    # 2. Run this test client
    python test_client.py --image path/to/xray.png

    # Or test batch prediction
    python test_client.py --batch image1.png image2.png image3.png
================================================================================
"""

import requests
import argparse
import json
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = "http://localhost:5000"

# ═══════════════════════════════════════════════════════════════════════════
# API CLIENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def check_health(base_url):
    """Check if the API server is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        data = response.json()
        
        print(f"\n{'='*80}")
        print("SERVER HEALTH CHECK")
        print(f"{'='*80}")
        print(f"Status: {data.get('status', 'unknown').upper()}")
        print(f"Model loaded: {'✓ YES' if data.get('model_loaded') else '✗ NO'}")
        print(f"GPU available: {'✓ YES' if data.get('gpu_available') else '✗ NO'}")
        print(f"Timestamp: {data.get('timestamp', 'unknown')}")
        print(f"{'='*80}\n")
        
        return data.get('model_loaded', False)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to server at {base_url}")
        print(f"  Make sure the server is running: python server_optimized.py\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return False


def get_model_info(base_url):
    """Get detailed model information"""
    try:
        response = requests.get(f"{base_url}/model-info", timeout=5)
        data = response.json()
        
        print(f"\n{'='*80}")
        print("MODEL INFORMATION")
        print(f"{'='*80}")
        print(f"Model name: {data.get('model_name', 'Unknown')}")
        print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
        
        perf = data.get('performance', {})
        print(f"\nPerformance Metrics:")
        print(f"  Test accuracy: {perf.get('test_accuracy', 0):.2f}%")
        print(f"  Sensitivity: {perf.get('sensitivity', 0):.2f}%")
        print(f"  Specificity: {perf.get('specificity', 0):.2f}%")
        print(f"  Precision: {perf.get('precision', 0):.2f}%")
        print(f"  F1-Score: {perf.get('f1_score', 0):.2f}%")
        
        config = data.get('configuration', {})
        print(f"\nConfiguration:")
        print(f"  Categories: {', '.join(config.get('categories', []))}")
        print(f"  Image size: {config.get('image_size', 'Unknown')}x{config.get('image_size', 'Unknown')}")
        print(f"  GPU used: {'✓ YES' if config.get('gpu_used') else '✗ NO'}")
        if config.get('gpu_used'):
            print(f"  GPU name: {config.get('gpu_name', 'Unknown')}")
        
        training = data.get('training_time', {})
        if training.get('total'):
            print(f"\nTraining Time:")
            print(f"  Total: {training['total']:.1f}s ({training['total']/60:.1f} min)")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR getting model info: {e}\n")


def predict_single(base_url, image_path):
    """Make a prediction on a single image"""
    if not Path(image_path).exists():
        print(f"\n✗ ERROR: Image file not found: {image_path}\n")
        return None
    
    try:
        print(f"\n{'='*80}")
        print(f"SINGLE IMAGE PREDICTION")
        print(f"{'='*80}")
        print(f"Image: {image_path}")
        print(f"Uploading and analyzing...\n")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{base_url}/predict",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✓ SUCCESS")
            print(f"{'-'*80}")
            print(f"Prediction: {data['prediction']}")
            print(f"Confidence: {data['confidence']*100:.2f}%")
            print(f"Risk Level: {data['risk_level']}")
            print(f"\nDetailed Probabilities:")
            for category, prob in data['probabilities'].items():
                bar_length = int(prob * 50)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"  {category:8s}: {bar} {prob*100:.2f}%")
            
            if 'processing_time_ms' in data:
                print(f"\nProcessing time: {data['processing_time_ms']:.1f}ms")
            print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
            print(f"{'='*80}\n")
            
            return data
        else:
            error_data = response.json()
            print(f"\n✗ ERROR: {error_data.get('error', 'Unknown error')}")
            if 'hint' in error_data:
                print(f"  Hint: {error_data['hint']}")
            print()
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to server at {base_url}\n")
        return None
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return None


def predict_batch(base_url, image_paths):
    """Make predictions on multiple images"""
    # Validate all files exist
    missing_files = [p for p in image_paths if not Path(p).exists()]
    if missing_files:
        print(f"\n✗ ERROR: File(s) not found:")
        for f in missing_files:
            print(f"  - {f}")
        print()
        return None
    
    try:
        print(f"\n{'='*80}")
        print(f"BATCH PREDICTION")
        print(f"{'='*80}")
        print(f"Number of images: {len(image_paths)}")
        print(f"Uploading and analyzing...\n")
        
        # Prepare files for upload
        files = [('images', (Path(p).name, open(p, 'rb'), 'image/png')) 
                 for p in image_paths]
        
        response = requests.post(
            f"{base_url}/predict-batch",
            files=files,
            timeout=60
        )
        
        # Close all file handles
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✓ SUCCESS")
            print(f"{'-'*80}")
            print(f"Total images: {data['total_images']}")
            print(f"Successful: {data['successful']}")
            print(f"Failed: {data['failed']}")
            print(f"Processing time: {data['processing_time_ms']:.1f}ms\n")
            
            # Show results for each image
            for result in data['results']:
                print(f"\n[{result['index']+1}] {result['filename']}")
                print(f"    Prediction: {result['prediction']}")
                print(f"    Confidence: {result['confidence']*100:.2f}%")
                print(f"    Risk Level: {result['risk_level']}")
                
                # Show probability bars
                probs = result['probabilities']
                for category, prob in probs.items():
                    bar_length = int(prob * 30)
                    bar = '█' * bar_length + '░' * (30 - bar_length)
                    print(f"      {category:8s}: {bar} {prob*100:.1f}%")
            
            # Show errors if any
            if data.get('errors'):
                print(f"\n{'='*80}")
                print("ERRORS:")
                for error in data['errors']:
                    print(f"  [{error['index']+1}] {error['filename']}: {error['error']}")
            
            print(f"\n{'='*80}\n")
            return data
        else:
            error_data = response.json()
            print(f"\n✗ ERROR: {error_data.get('error', 'Unknown error')}\n")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to server at {base_url}\n")
        return None
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='TB Detection API Test Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check server health
  python test_client.py --health

  # Get model information
  python test_client.py --info

  # Predict single image
  python test_client.py --image xray.png

  # Predict multiple images (batch)
  python test_client.py --batch image1.png image2.png image3.png

  # Use custom server URL
  python test_client.py --url http://192.168.1.100:8080 --image xray.png
        """
    )
    
    parser.add_argument('--url', type=str, default=API_BASE_URL,
                       help=f'API server URL (default: {API_BASE_URL})')
    parser.add_argument('--health', action='store_true',
                       help='Check server health')
    parser.add_argument('--info', action='store_true',
                       help='Get model information')
    parser.add_argument('--image', type=str,
                       help='Path to single X-ray image for prediction')
    parser.add_argument('--batch', type=str, nargs='+',
                       help='Paths to multiple images for batch prediction')
    parser.add_argument('--json', action='store_true',
                       help='Output raw JSON response')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print(f"\n{'='*80}")
        print("QUICK START:")
        print(f"{'='*80}")
        print("1. Start the server: python server_optimized.py")
        print("2. Test the server: python test_client.py --health")
        print("3. Make a prediction: python test_client.py --image xray.png")
        print(f"{'='*80}\n")
        sys.exit(0)
    
    base_url = args.url
    
    # Execute requested action
    if args.health:
        check_health(base_url)
    
    if args.info:
        if not check_health(base_url):
            sys.exit(1)
        get_model_info(base_url)
    
    if args.image:
        if not check_health(base_url):
            sys.exit(1)
        result = predict_single(base_url, args.image)
        if args.json and result:
            print(json.dumps(result, indent=2))
    
    if args.batch:
        if not check_health(base_url):
            sys.exit(1)
        result = predict_batch(base_url, args.batch)
        if args.json and result:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()