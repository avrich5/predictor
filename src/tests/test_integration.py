import unittest
import numpy as np
from src.data.ohlcv_processor import OHLCVProcessor
from src.core.markov_predictor import AdvancedMarkovPredictor
from src.core.predictor_manager import PredictorManager
from src.config.predictor_config import PredictorConfig

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты для проверки взаимодействия компонентов"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.processor = OHLCVProcessor()
        self.manager = PredictorManager()
        self.symbol = "BTCUSDT"
        self.predictor_id = "test_predictor"
        
        # Создаем тестовые OHLCV данные
        self.test_klines = []
        for i in range(100):
            base_price = 8000 + i * 10
            self.test_klines.append([
                1499040000000 + i * 3600000,  # Открытие свечи
                str(base_price),              # Open price
                str(base_price + 50),         # High price
                str(base_price - 50),         # Low price
                str(base_price + 20),         # Close price
                "10.5",                       # Volume
                1499043600000 + i * 3600000,  # Закрытие свечи
                "85050.0",                    # Quote asset volume
                120,                          # Количество сделок
                "5.5",                        # Taker buy base asset volume
                "44680.0",                    # Taker buy quote asset volume
                "0"                           # Игнорируется
            ])
    
    def test_processor_to_predictor_workflow(self):
        """Тест рабочего процесса от обработчика данных к предиктору"""
        # Добавляем данные в процессор
        self.processor.add_klines(self.symbol, self.test_klines)
        
        # Получаем массив цен
        prices = self.processor.get_prices(self.symbol, 'close')
        
        # Создаем предиктор
        config = PredictorConfig(
            window_size=50,
            prediction_depth=5,
            min_confidence=0.55,
            state_length=3,
            significant_change_pct=0.4,
            use_weighted_window=False
        )
        predictor = AdvancedMarkovPredictor(config)
        
        # Запускаем предиктор на данных
        results = predictor.run_on_data(prices, verbose=False)
        
        # Проверяем, что предсказания были сделаны
        self.assertGreater(len(results), 0)
        self.assertGreater(predictor.total_predictions, 0)
    
    def test_manager_with_ohlcv_data(self):
        """Тест работы менеджера предикторов с OHLCV данными"""
        # Инициализируем предиктор через менеджер
        config_dict = {
            "window_size": 50,
            "prediction_depth": 5,
            "min_confidence": 0.55,
            "state_length": 3,
            "significant_change_pct": 0.4,
            "use_weighted_window": False,
            "price_source": "close",
            "price_calculation": "single"
        }
        self.manager.initialize_predictor(self.predictor_id, config_dict, {})
        
        # Отправляем данные в предиктор через менеджер
        for kline in self.test_klines:
            prediction = self.manager.make_prediction(self.predictor_id, kline)
            # Проверяем, что получили предсказание
            self.assertIsNotNone(prediction)
            self.assertIn('prediction', prediction)
        
        # Получаем статус предиктора
        status = self.manager.get_predictor_status(self.predictor_id)
        
        # Проверяем, что статус содержит ожидаемые поля
        self.assertIsNotNone(status)
        self.assertIn('statistics', status)
        self.assertIn('points_processed', status['statistics'])
        self.assertEqual(status['statistics']['points_processed'], len(self.test_klines))

if __name__ == '__main__':
    unittest.main()