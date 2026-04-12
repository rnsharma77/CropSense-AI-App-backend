"""
CropSense AI — ML Prediction (Vercel Serverless Python Function)
This function runs on Vercel and provides ML predictions via HTTP API.
Local execution: python -m http.server (then POST to http://localhost:8000/api/ml_predict)
Vercel deployment: Automatically deployed as serverless function
"""

import json
import base64
import sys
import io
from pathlib import Path

torch = None
nn = None
transforms = None
models = None
Image = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
except ImportError as e:
    print(f"Warning: PyTorch imports failed: {e}", file=sys.stderr)

# Model paths: prioritize ml_model folder for Vercel, fallback to ml folder
ALT_MODEL_DIR = Path(__file__).parent.parent / 'ml_model' / 'models'
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / 'ml' / 'models'
MODEL_DIR = ALT_MODEL_DIR if ALT_MODEL_DIR.exists() else DEFAULT_MODEL_DIR
MODEL_PATH = MODEL_DIR / 'cropsense_model.pth'
DEVICE = torch.device('cpu') if torch is not None else None

# Global model cache (persists across invocations in Vercel)
_model_cache = None
_checkpoint_cache = None


def resolve_idx_to_class(raw_idx_to_class):
    """Convert dict keys to integers"""
    if not raw_idx_to_class:
        return {}
    resolved = {}
    for key, value in raw_idx_to_class.items():
        try:
            resolved[int(key)] = value
        except (TypeError, ValueError):
            continue
    return resolved


def load_model():
    """Load model from checkpoint (cached globally)"""
    global _model_cache, _checkpoint_cache

    if torch is None or nn is None or models is None:
        return None, None
    
    if _model_cache is not None:
        return _model_cache, _checkpoint_cache
    
    if not MODEL_PATH.exists():
        return None, None
    
    try:
        ck = torch.load(MODEL_PATH, map_location=DEVICE)
        m = models.mobilenet_v3_small(weights=None)
        inf = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(inf, ck['num_classes'])
        m.load_state_dict(ck['model_state'])
        m.eval()
        
        _model_cache = m
        _checkpoint_cache = ck
        return m, ck
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None


def predict(model, ck, img_bytes):
    """Run inference on image bytes"""
    try:
        sz = ck.get('img_size', 224)
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = tf(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            probs = torch.softmax(model(tensor), 1)[0]
        
        idx_to_class = resolve_idx_to_class(ck.get('idx_to_class'))
        k = min(3, len(idx_to_class))
        vals, idxs = torch.topk(probs, k)
        
        preds = [
            {
                'disease': idx_to_class.get(i.item(), f'class_{i.item()}'),
                'probability': round(p.item(), 4),
            }
            for p, i in zip(vals, idxs)
        ]
        
        top = preds[0]
        return {
            'success': True,
            'source': 'local_ml',
            'disease': top['disease'],
            'confidence': top['probability'],
            'severityScore': int(top['probability'] * 100),
            'severity': 'High' if top['probability'] > 0.7 else 'Medium' if top['probability'] > 0.4 else 'Low',
            'topPredictions': preds,
            'modelAccuracy': ck.get('val_acc', 0),
        }
    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }


def sanitize_base64(value):
    return str(value or '').strip().split(',', 1)[-1]


def handler(request):
    """
    Vercel serverless handler function.
    Request format: POST with JSON body containing 'imageBase64'
    """
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'success': False, 'error': 'Method not allowed'})
        }
    
    try:
        # Parse request body
        try:
            body = json.loads(request.body) if isinstance(request.body, str) else request.body
        except:
            body = request.body if isinstance(request.body, dict) else {}
        
        image_base64 = body.get('imageBase64', '')
        if not image_base64:
            return {
                'statusCode': 400,
                'body': json.dumps({'success': False, 'error': 'imageBase64 is required'})
            }
        
        # Decode base64 image
        try:
            img_bytes = base64.b64decode(sanitize_base64(image_base64))
        except Exception as e:
            return {
                'statusCode': 400,
                'body': json.dumps({'success': False, 'error': f'Invalid base64: {str(e)}'})
            }
        
        # Load model
        model, ck = load_model()
        if model is None:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': 'Model not loaded. Check Python ML dependencies and trained model files.'
                })
            }
        
        # Run prediction
        result = predict(model, ck, img_bytes)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(result)
        }
    
    except Exception as e:
        print(f"Handler error: {e}", file=sys.stderr)
        return {
            'statusCode': 500,
            'body': json.dumps({'success': False, 'error': f'Server error: {str(e)}'})
        }


# For local testing
if __name__ == '__main__':
    # Simple HTTP server for local testing
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            class MockRequest:
                pass
            
            req = MockRequest()
            req.method = 'POST'
            req.body = body.decode('utf-8')
            
            response = handler(req)
            
            self.send_response(response['statusCode'])
            for key, value in response.get('headers', {}).items():
                self.send_header(key, value)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response['body'].encode('utf-8'))
        
        def log_message(self, format, *args):
            pass  # Suppress default logging
    
    print("Starting local ML predict server on http://localhost:8001/api/ml_predict")
    server = HTTPServer(('localhost', 8001), RequestHandler)
    server.serve_forever()
