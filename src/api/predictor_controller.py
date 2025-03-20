from src.core.predictor_manager import PredictorManager

class PredictorController:
    """Controller for handling API requests related to predictors"""
    
    def __init__(self):
        self.manager = PredictorManager()
    
    def initialize(self, data):
        """Initialize a new predictor or reset an existing one"""
        predictor_id = data.get('predictor_id', 'default')
        config = data.get('config', {})
        filters = data.get('filters', {})
        
        success = self.manager.initialize_predictor(predictor_id, config, filters)
        
        if success:
            return {
                'status': 'success',
                'message': 'Predictor initialized',
                'predictor_id': predictor_id
            }
        else:
            return {
                'status': 'error',
                'error_code': 'initialization_failed',
                'message': 'Failed to initialize predictor'
            }
    
    def feed(self, predictor_id, data):
        """Feed new data to the predictor and get a prediction"""
        kline = data.get('kline', {})
        additional_data = data.get('additional_data', {})
        
        # Проверяем существование предиктора
        if not self.manager.predictor_exists(predictor_id):
            return {
                'status': 'error',
                'error_code': 'predictor_not_found',
                'message': f'Predictor with ID {predictor_id} not found'
            }
        
        # Получаем предсказание
        prediction_result = self.manager.make_prediction(predictor_id, kline, additional_data)
        
        return {
            'status': 'success',
            'prediction': prediction_result['prediction'],
            'statistics': prediction_result['statistics']
        }
    
    def get_status(self, predictor_id):
        """Get the current status of the predictor"""
        # Проверяем существование предиктора
        if not self.manager.predictor_exists(predictor_id):
            return {
                'status': 'error',
                'error_code': 'predictor_not_found',
                'message': f'Predictor with ID {predictor_id} not found'
            }
        
        status_data = self.manager.get_predictor_status(predictor_id)
        
        return {
            'status': 'success',
            **status_data
        }
    
    def batch_process(self, predictor_id, data):
        """Process a batch of historical data"""
        # Проверяем существование предиктора
        if not self.manager.predictor_exists(predictor_id):
            return {
                'status': 'error',
                'error_code': 'predictor_not_found',
                'message': f'Predictor with ID {predictor_id} not found'
            }
        
        # Получаем klines из данных
        klines = data.get('klines', [])
        
        # Обрабатываем данные через менеджер
        result = self.manager.batch_process(predictor_id, data)
        
        return {
            'status': 'success',
            'message': f'Processed {len(klines)} klines',
            **result
        }
    
    def validate_predictions(self, predictor_id):
        """Validate pending predictions"""
        # Проверяем существование предиктора
        if not self.manager.predictor_exists(predictor_id):
            return {
                'status': 'error',
                'error_code': 'predictor_not_found',
                'message': f'Predictor with ID {predictor_id} not found'
            }
        
        # Получаем результаты валидации
        result = self.manager.validate_predictions(predictor_id)
        
        return {
            'status': 'success',
            'validated': result['validated'],
            'correct': result['correct'],
            'incorrect': result['incorrect'],
            'success_rate': result['success_rate']
        }

    
    def generate_report(self, predictor_id, format_type='json', from_date=None, to_date=None):
        """Generate a report about the predictor's performance"""
        # Проверяем существование предиктора
        if not self.manager.predictor_exists(predictor_id):
            return {
                'status': 'error',
                'error_code': 'predictor_not_found',
                'message': f'Predictor with ID {predictor_id} not found'
            }
        
        # Получаем предиктор из менеджера
        predictor_data = self.manager.predictors.get(predictor_id)
        predictor = predictor_data['predictor']
        
        # Получаем список результатов для отчета
        results = []
        for idx, stat in predictor.point_statistics.items():
            results.append({
                'index': idx,
                'success_rate': stat['success_rate'] * 100 if 'success_rate' in stat else 0
            })
        
        # Получаем статистику по состояниям
        state_stats = []
        if hasattr(predictor, 'state_statistics'):
            for state, stats in predictor.state_statistics.items():
                success_rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                state_stats.append({
                    'state': str(state),
                    'occurrences': stats['total'],
                    'correct': stats['correct'],
                    'success_rate': success_rate
                })
        
        # Сортируем состояния по количеству вхождений
        state_stats = sorted(state_stats, key=lambda x: x['occurrences'], reverse=True)
        
        # Формируем отчет
        report = {
            'status': 'success',
            'report': {
                'general_statistics': {
                    'total_predictions': predictor.total_predictions if hasattr(predictor, 'total_predictions') else 0,
                    'correct_predictions': predictor.correct_predictions if hasattr(predictor, 'correct_predictions') else 0,
                    'success_rate': predictor.success_rate * 100 if hasattr(predictor, 'success_rate') else 0,
                    'data_points_processed': len(predictor_data['data_buffer']) if 'data_buffer' in predictor_data else 0
                },
                'state_statistics': state_stats[:10],  # Топ 10 состояний
                'prediction_history': results[-100:]  # Последние 100 предсказаний
            }
        }
        
        # Если требуется формат Markdown
        if format_type.lower() == 'markdown':
            # Преобразуем отчет в Markdown (упрощенная версия)
            markdown = f"""
    # Отчет о работе предиктора {predictor_id}

    ## Общая статистика
    - Всего предсказаний: {report['report']['general_statistics']['total_predictions']}
    - Правильных предсказаний: {report['report']['general_statistics']['correct_predictions']}
    - Успешность: {report['report']['general_statistics']['success_rate']:.2f}%
    - Обработано точек данных: {report['report']['general_statistics']['data_points_processed']}

    ## Топ состояний по частоте
    """
            
            if state_stats:
                markdown += "| Состояние | Вхождений | Верных | Успешность (%) |\n"
                markdown += "|-----------|-----------|--------|---------------|\n"

    # Другие методы для остальных эндпоинтов...