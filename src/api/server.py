from flask import Flask, jsonify
from src.api.routes import register_routes

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Регистрируем все маршруты
    register_routes(app)
    
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'ok'})
    
    return app