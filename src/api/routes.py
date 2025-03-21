from flask import Blueprint, request, jsonify
from src.api.predictor_controller import PredictorController
from src.data.binance_client import BinanceClient

# Initialize controller once
controller = PredictorController()

# Create blueprints
predictor_bp = Blueprint('predictor', __name__, url_prefix='/api/predictor')
data_bp = Blueprint('data', __name__, url_prefix='/api/data')

@predictor_bp.route('/initialize', methods=['POST'])
def initialize():
    """Initialize a new predictor or reset an existing one"""
    data = request.json
    if not data:
        return jsonify({
            'status': 'error',
            'error_code': 'invalid_request',
            'message': 'No JSON data provided'
        }), 400
    result = controller.initialize(data)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/feed', methods=['POST'])
def feed(predictor_id):
    """Feed new data to the predictor and get a prediction"""
    data = request.json
    if not data:
        return jsonify({
            'status': 'error',
            'error_code': 'invalid_request',
            'message': 'No JSON data provided'
        }), 400
    result = controller.feed(predictor_id, data)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/status', methods=['GET'])
def status(predictor_id):
    """Get the current status of the predictor"""
    result = controller.get_status(predictor_id)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/validate', methods=['POST'])
def validate(predictor_id):
    """Validate pending predictions"""
    result = controller.validate_predictions(predictor_id)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/report', methods=['GET'])
def get_report(predictor_id):
    """Generate a report about the predictor's performance"""
    format_type = request.args.get('format', 'json')
    from_date = request.args.get('from')
    to_date = request.args.get('to')
    result = controller.generate_report(predictor_id, format_type, from_date, to_date)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/config', methods=['PATCH'])
def update_config(predictor_id):
    """Update the predictor's configuration"""
    data = request.json
    if not data:
        return jsonify({
            'status': 'error',
            'error_code': 'invalid_request',
            'message': 'No JSON data provided'
        }), 400
    result = controller.update_config(predictor_id, data)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/batch', methods=['POST'])
def batch_process(predictor_id):
    """Process a batch of historical data"""
    data = request.json
    if not data:
        return jsonify({
            'status': 'error',
            'error_code': 'invalid_request',
            'message': 'No JSON data provided'
        }), 400
    result = controller.batch_process(predictor_id, data)
    return jsonify(result)

@predictor_bp.route('/<predictor_id>/visualize', methods=['GET'])
def visualize(predictor_id):
    """Generate a visualization of the predictor's results"""
    viz_type = request.args.get('type', 'price_chart')
    format_type = request.args.get('format', 'json')
    result = controller.visualize(predictor_id, viz_type, format_type)
    
    if format_type == 'svg' and not isinstance(result, dict):
        return result, 200, {'Content-Type': 'image/svg+xml'}
    return jsonify(result)

@data_bp.route('/binance/klines', methods=['GET'])
def get_binance_klines():
    """Get historical klines from Binance"""
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '1h')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    limit = request.args.get('limit', 500)
    predictor_id = request.args.get('predictor_id')
    
    # Преобразуем строковые параметры в целые числа
    if start_time:
        start_time = int(start_time)
    if end_time:
        end_time = int(end_time)
    if limit:
        limit = int(limit)
    
    # Получаем данные с Binance
    client = BinanceClient()
    klines = client.get_klines(symbol, interval, start_time, end_time, limit)
    
    # Если указан predictor_id, отправляем данные в предиктор
    if predictor_id and controller.manager.predictor_exists(predictor_id):
        # Используем batch_process для обработки данных
        batch_result = controller.batch_process(predictor_id, {'klines': klines})
        
        return jsonify({
            'status': 'success',
            'klines_fetched': len(klines),
            'symbol': symbol,
            'interval': interval,
            'start_time': start_time,
            'end_time': end_time,
            'batch_result': batch_result
        })
    
    # Иначе просто возвращаем данные
    return jsonify({
        'status': 'success',
        'klines_fetched': len(klines),
        'symbol': symbol,
        'interval': interval,
        'start_time': start_time,
        'end_time': end_time,
        'klines': klines
    })

def register_routes(app):
    """Register all API routes"""
    # Регистрируем Blueprint для предиктора
    app.register_blueprint(predictor_bp)
    
    # Регистрируем Blueprint для работы с данными
    app.register_blueprint(data_bp)
    