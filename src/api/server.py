from flask import Flask, jsonify, redirect, url_for
from src.api.routes import register_routes

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Add root route to redirect to health check
    @app.route('/')
    def index():
        """Redirect root to API documentation or health check"""
        return jsonify({
            'status': 'ok',
            'name': 'Markov Predictor Service',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'predictor': {
                    'initialize': '/api/predictor/initialize',
                    'feed': '/api/predictor/{predictor_id}/feed',
                    'status': '/api/predictor/{predictor_id}/status',
                    'report': '/api/predictor/{predictor_id}/report',
                    'validate': '/api/predictor/{predictor_id}/validate',
                    'batch': '/api/predictor/{predictor_id}/batch',
                    'visualize': '/api/predictor/{predictor_id}/visualize',
                    'config': '/api/predictor/{predictor_id}/config'
                },
                'data': {
                    'binance_klines': '/api/data/binance/klines'
                }
            },
            'documentation': 'See README.md for more information'
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint to verify the server is running"""
        return jsonify({'status': 'ok'})
    
    # Register API routes
    register_routes(app)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5001)
    