from src.core.markov_predictor import AdvancedMarkovPredictor
from src.config.predictor_config import PredictorConfig
import time
import numpy as np
from collections import defaultdict

class PredictorManager:
    """
    Manages multiple predictor instances and their configurations
    """
    
    def __init__(self):
        self.predictors = {}
    
    def predictor_exists(self, predictor_id):
        """Check if a predictor with the given ID exists"""
        return predictor_id in self.predictors
    
    def initialize_predictor(self, predictor_id, config_dict, filters_dict):
        """Initialize a new predictor or reset an existing one"""
        try:
            # Создаем объект конфигурации
            config = PredictorConfig(**config_dict)
            
            # Создаем предиктор
            predictor = AdvancedMarkovPredictor(config)
            
            # Применяем фильтры, если они есть
            if filters_dict:
                predictor.set_filters(filters_dict)
            
            # Сохраняем предиктор в словаре
            self.predictors[predictor_id] = {
                'predictor': predictor,
                'config': config,
                'filters': filters_dict,
                'data_buffer': [],  # Буфер для хранения последних данных
                'pending_predictions': {}  # Словарь ожидающих проверки предсказаний
            }
            
            return True
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            return False
    
    def make_prediction(self, predictor_id, kline, additional_data=None):
        """Make a prediction using the specified predictor"""
        predictor_data = self.predictors.get(predictor_id)
        if not predictor_data:
            return None
        
        predictor = predictor_data['predictor']
        config = predictor_data['config']
        
        # Извлекаем необходимые данные из kline
        # Если kline - это список (формат Binance), преобразуем его в словарь
        if isinstance(kline, list):
            kline_dict = {
                'open_time': kline[0],
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': kline[6],
                'quote_asset_volume': float(kline[7]),
                'number_of_trades': kline[8],
                'taker_buy_base_asset_volume': float(kline[9]),
                'taker_buy_quote_asset_volume': float(kline[10])
            }
        else:
            kline_dict = kline
        
        # Определяем, какую цену использовать (по умолчанию close)
        price_source = config.price_source if hasattr(config, 'price_source') else 'close'
        price = float(kline_dict.get(price_source, kline_dict.get('close')))
        
        # Добавляем данные в буфер
        predictor_data['data_buffer'].append({
            'price': price,
            'kline': kline_dict,
            'additional_data': additional_data
        })
        
        # Если буфер слишком большой, удаляем старые данные
        max_buffer_size = max(1000, config.window_size * 2)
        if len(predictor_data['data_buffer']) > max_buffer_size:
            predictor_data['data_buffer'] = predictor_data['data_buffer'][-max_buffer_size:]
        
        # Извлекаем массив цен из буфера данных
        prices = np.array([item['price'] for item in predictor_data['data_buffer']])
        
        # Делаем предсказание, если у нас достаточно данных
        if len(prices) >= config.window_size:
            # Вызываем метод predict_at_point для текущей точки (последней в массиве)
            prediction_result = predictor.predict_at_point(prices, len(prices) - 1)
            
            # Формируем ID для этого предсказания (используем timestamp)
            prediction_id = str(int(time.time() * 1000))
            
            # Сохраняем предсказание для будущей валидации
            predictor_data['pending_predictions'][prediction_id] = {
                'timestamp': int(time.time()),
                'price': price,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'state': prediction_result.get('state'),
                'target_time': int(time.time()) + (config.prediction_depth * 60)  # Примерное время для проверки (зависит от таймфрейма)
            }
            
            # Формируем ответ
            response = {
                'prediction': {
                    'signal': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'state': prediction_result.get('state'),
                    'state_occurrences': prediction_result.get('state_occurrences', 0),
                    'state_success_rate': self._get_state_success_rate(predictor, prediction_result.get('state'))
                },
                'statistics': {
                    'total_predictions': predictor.total_predictions,
                    'correct_predictions': predictor.correct_predictions,
                    'success_rate': predictor.success_rate * 100 if hasattr(predictor, 'success_rate') else 0,
                    'points_processed': len(predictor_data['data_buffer'])
                }
            }
        else:
            # Если недостаточно данных, возвращаем "не знаю"
            response = {
                'prediction': {
                    'signal': 0,  # 0 = не знаю
                    'confidence': 0.0,
                    'state': None,
                    'state_occurrences': 0,
                    'state_success_rate': 0
                },
                'statistics': {
                    'total_predictions': predictor.total_predictions if hasattr(predictor, 'total_predictions') else 0,
                    'correct_predictions': predictor.correct_predictions if hasattr(predictor, 'correct_predictions') else 0,
                    'success_rate': predictor.success_rate * 100 if hasattr(predictor, 'success_rate') else 0,
                    'points_processed': len(predictor_data['data_buffer'])
                }
            }
        
        return response

    def _get_state_success_rate(self, predictor, state):
        """Получает успешность для конкретного состояния"""
        if not state or not hasattr(predictor, 'state_statistics'):
            return 0
        
        state_str = str(state)
        stats = predictor.state_statistics.get(state)
        if not stats or stats['total'] == 0:
            return 0
    
        return (stats['correct'] / stats['total']) * 100
    
    def get_predictor_status(self, predictor_id):
        """Get the current status of the specified predictor"""
        predictor_data = self.predictors.get(predictor_id)
        if not predictor_data:
            return None
        
        predictor = predictor_data['predictor']
        config = predictor_data['config']
        
        # Собираем статистику по состояниям
        state_stats = []
        if hasattr(predictor, 'state_statistics'):
            for state, stats in predictor.state_statistics.items():
                success_rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                state_stats.append({
                    'state': state,
                    'occurrences': stats['total'],
                    'correct': stats['correct'],
                    'success_rate': success_rate
                })
            
            # Сортируем по количеству вхождений (от большего к меньшему)
            state_stats.sort(key=lambda x: x['occurrences'], reverse=True)
        
        # Собираем последние предсказания
        recent_predictions = []
        for pred_id, prediction in list(predictor_data['pending_predictions'].items())[-10:]:
            recent_predictions.append({
                'id': pred_id,
                'timestamp': prediction['timestamp'],
                'price': prediction['price'],
                'signal': prediction['prediction'],
                'confidence': prediction['confidence'],
                'target_time': prediction['target_time']
            })
        
        # Формируем статус
        status_data = {
            'config': vars(config),
            'statistics': {
                'total_predictions': predictor.total_predictions if hasattr(predictor, 'total_predictions') else 0,
                'correct_predictions': predictor.correct_predictions if hasattr(predictor, 'correct_predictions') else 0,
                'success_rate': predictor.success_rate * 100 if hasattr(predictor, 'success_rate') else 0,
                'coverage_percentage': self._calculate_coverage(predictor) if hasattr(predictor, 'total_predictions') else 0,
                'points_processed': len(predictor_data['data_buffer']),
                'pending_predictions': len(predictor_data['pending_predictions'])
            },
            'state_statistics': state_stats[:10],  # Топ-10 состояний
            'recent_predictions': recent_predictions
        }
        
        return status_data

    def _calculate_coverage(self, predictor):
        """Calculate prediction coverage percentage"""
        if not hasattr(predictor, 'total_predictions') or not hasattr(predictor, 'point_statistics'):
            return 0
        
        if not predictor.point_statistics:
            return 0
        
        total_points = max(predictor.point_statistics.keys()) + 1 if predictor.point_statistics else 0
        if total_points == 0:
            return 0
        
        return (predictor.total_predictions / total_points) * 100
    
    def validate_predictions(self, predictor_id, actual_price=None):
        """
        Validates pending predictions using the current price
        
        Parameters:
        predictor_id (str): ID of the predictor
        actual_price (float, optional): The actual price to use for validation
        
        Returns:
        dict: Validation results
        """
        predictor_data = self.predictors.get(predictor_id)
        if not predictor_data:
            return None
        
        predictor = predictor_data['predictor']
        config = predictor_data['config']
        pending_predictions = predictor_data['pending_predictions']
        
        # Если нет ожидающих предсказаний, возвращаем пустой результат
        if not pending_predictions:
            return {
                'validated': 0,
                'correct': 0,
                'incorrect': 0,
                'success_rate': 0
            }
        
        # Если цена не передана, берем последнюю из буфера
        if actual_price is None and predictor_data['data_buffer']:
            actual_price = predictor_data['data_buffer'][-1]['price']
        
        # Если и буфер пуст, нечего валидировать
        if actual_price is None:
            return {
                'validated': 0,
                'correct': 0,
                'incorrect': 0,
                'success_rate': 0
            }
        
        current_time = int(time.time())
        validated = 0
        correct = 0
        incorrect = 0
        
        # Проверяем все ожидающие предсказания
        to_remove = []
        for pred_id, prediction in pending_predictions.items():
            # Проверяем только предсказания, которые достигли своего target_time
            if prediction['target_time'] <= current_time:
                validated += 1
                
                # Определяем фактический результат
                initial_price = prediction['price']
                pct_change = (actual_price - initial_price) / initial_price if initial_price != 0 else 0
                
                # Определяем фактическое направление движения с учетом порога significant_change_pct
                if pct_change > config.significant_change_pct:
                    actual_outcome = 1  # Значимый рост
                elif pct_change < -config.significant_change_pct:
                    actual_outcome = 2  # Значимое падение
                else:
                    actual_outcome = 0  # Незначительное изменение
                
                # Проверяем, было ли предсказание верным
                prediction_signal = prediction['prediction']
                if prediction_signal == 0 or actual_outcome == 0:
                    # Если предсказание "не знаю" или результат незначителен, не считаем
                    pass
                elif prediction_signal == actual_outcome:
                    correct += 1
                    # Обновляем статистику предиктора
                    predictor.total_predictions += 1
                    predictor.correct_predictions += 1
                else:
                    incorrect += 1
                    # Обновляем статистику предиктора
                    predictor.total_predictions += 1
                
                # Обновляем общую успешность
                if hasattr(predictor, 'total_predictions') and predictor.total_predictions > 0:
                    predictor.success_rate = predictor.correct_predictions / predictor.total_predictions
                
                # Отмечаем это предсказание для удаления
                to_remove.append(pred_id)
        
        # Удаляем проверенные предсказания
        for pred_id in to_remove:
            del pending_predictions[pred_id]
        
        # Формируем результат
        result = {
            'validated': validated,
            'correct': correct,
            'incorrect': incorrect,
            'success_rate': (correct / validated * 100) if validated > 0 else 0
        }
        
        return result    
    
    def batch_process(self, predictor_id, data):
        predictor_data = self.predictors.get(predictor_id)
        if not predictor_data:
            return None
        
        predictor = predictor_data['predictor']
        config = predictor_data['config']
        
        klines = data.get('klines', [])
        options = data.get('options', {})
        
        # Если нужно сбросить статистику
        if options.get('reset_statistics', False):
            predictor.total_predictions = 0
            predictor.correct_predictions = 0
            predictor.success_rate = 0.0
            predictor.point_statistics = {}
            predictor.state_statistics = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Обрабатываем данные последовательно
        for kline in klines:
            self.make_prediction(predictor_id, kline)
        
        # Формируем результат
        result = {
            'status': 'success',
            'points_processed': len(klines),
            'statistics': {
                'total_predictions': predictor.total_predictions if hasattr(predictor, 'total_predictions') else 0,
                'correct_predictions': predictor.correct_predictions if hasattr(predictor, 'correct_predictions') else 0,
                'success_rate': predictor.success_rate * 100 if hasattr(predictor, 'success_rate') else 0,
                'buffer_size': len(predictor_data['data_buffer'])
            }
        }
        
        return result
    
    # Другие методы для остальных функций...