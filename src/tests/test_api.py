import json
import unittest
from src.app import create_app

class TestAPI(unittest.TestCase):
    """Тесты для API эндпоинтов"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.testing = True
        
        # Создаем тестовый предиктор
        self.test_predictor_id = "test_predictor"
        response = self.client.post(
            '/api/predictor/initialize',
            data=json.dumps({
                "predictor_id": self.test_predictor_id,
                "config": {
                    "window_size": 50,
                    "prediction_depth": 5,
                    "min_confidence": 0.55,
                    "state_length": 3,
                    "significant_change_pct": 0.4,
                    "use_weighted_window": False
                }
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
    def test_health_check(self):
        """Тест проверки состояния API"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
    
    def test_initialize_predictor(self):
        """Тест инициализации предиктора"""
        response = self.client.post(
            '/api/predictor/initialize',
            data=json.dumps({
                "predictor_id": "new_predictor",
                "config": {
                    "window_size": 100,
                    "prediction_depth": 10,
                    "min_confidence": 0.6,
                    "state_length": 3,
                    "significant_change_pct": 0.5,
                    "use_weighted_window": True,
                    "weight_decay": 0.9
                }
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['predictor_id'], 'new_predictor')
    
    def test_get_status(self):
        """Тест получения статуса предиктора"""
        response = self.client.get(f'/api/predictor/{self.test_predictor_id}/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('config', data)
        self.assertIn('statistics', data)
    
    def test_feed_data(self):
        """Тест отправки данных в предиктор"""
        # Формат Binance kline
        kline = [
            1499040000000,      # Открытие свечи
            "8100.0",           # Open price
            "8150.0",           # High price
            "8050.0",           # Low price
            "8120.0",           # Close price
            "10.5",             # Volume
            1499043600000,      # Закрытие свечи
            "85050.0",          # Quote asset volume
            120,                # Количество сделок
            "5.5",              # Taker buy base asset volume
            "44680.0",          # Taker buy quote asset volume
            "0"                 # Игнорируется
        ]
        
        response = self.client.post(
            f'/api/predictor/{self.test_predictor_id}/feed',
            data=json.dumps({
                "kline": kline
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data)
        self.assertIn('statistics', data)
    
    def test_batch_process(self):
        """Тест пакетной обработки данных"""
        # Создаем несколько свечей
        klines = []
        for i in range(10):
            price = 8000 + i * 10
            klines.append([
                1499040000000 + i * 3600000,
                str(price),
                str(price + 50),
                str(price - 50),
                str(price + 20),
                "10.5",
                1499043600000 + i * 3600000,
                "85050.0",
                120,
                "5.5",
                "44680.0",
                "0"
            ])
        
        response = self.client.post(
            f'/api/predictor/{self.test_predictor_id}/batch',
            data=json.dumps({
                "klines": klines,
                "options": {
                    "reset_statistics": True
                }
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], f'Processed {len(klines)} klines')
    
    def test_validate_predictions(self):
        """Тест валидации предсказаний"""
        # Сначала отправляем данные для создания предсказаний
        self.test_batch_process()
        
        # Теперь валидируем
        response = self.client.post(
            f'/api/predictor/{self.test_predictor_id}/validate',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('validated', data)

def setUp(self):
    """Настройка перед каждым тестом"""
    # Create the app
    self.app = create_app()
    self.client = self.app.test_client()
    self.client.testing = True
    
    # Set up a predictor ID for testing
    self.test_predictor_id = "test_predictor"
    
    # Initialize a test predictor
    response = self.client.post(
        '/api/predictor/initialize',
        json={
            "predictor_id": self.test_predictor_id,
            "config": {
                "window_size": 50,
                "prediction_depth": 5,
                "min_confidence": 0.55,
                "state_length": 3,
                "significant_change_pct": 0.4,
                "use_weighted_window": False
            }
        }
    )
    
    # Ensure the predictor was created successfully
    response_data = json.loads(response.data)
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response_data.get('status'), 'success')
if __name__ == '__main__':
    unittest.main()