import unittest
import numpy as np
from src.core.markov_predictor import AdvancedMarkovPredictor
from src.config.predictor_config import PredictorConfig

class TestPredictor(unittest.TestCase):
    """Тесты для работы предиктора"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.config = PredictorConfig(
            window_size=50,
            prediction_depth=5,
            min_confidence=0.55,
            state_length=3,
            significant_change_pct=0.4,
            use_weighted_window=False
        )
        self.predictor = AdvancedMarkovPredictor(self.config)
    
    def test_determine_movement(self):
        """Тест определения направления движения"""
        # Значимый рост (больше 0.4%)
        result = self.predictor._determine_movement(100, 100.5)
        self.assertEqual(result, 1)
        
        # Значимое падение (меньше -0.4%)
        result = self.predictor._determine_movement(100, 99.5)
        self.assertEqual(result, 2)
        
        # Незначительное изменение
        result = self.predictor._determine_movement(100, 100.2)
        self.assertEqual(result, 0)
    
    def test_get_state(self):
        """Тест получения состояния"""
        # Создаем тестовые данные
        prices = np.array([100, 101, 99, 100, 101])
        
        # Получаем состояние в точке 3 (после 3-х движений)
        state = self.predictor._get_state(prices, 3)
        self.assertEqual(state, (1, 2, 1))  # рост, падение, рост
    
    def test_predict_at_point(self):
        """Тест предсказания в точке"""
        # Создаем тестовые данные - периодическая последовательность
        prices = np.array([100] * 10 + [101] * 10 + [99] * 10 + [100] * 10 + [101] * 10)
        
        # Делаем предсказание после достаточного количества данных
        result = self.predictor.predict_at_point(prices, 40)
        
        # Проверяем, что предсказание сделано
        self.assertIn('prediction', result)
        
    def test_run_on_data(self):
        """Тест запуска на данных"""
        # Создаем тестовые данные
        prices = np.array([100 + i * 0.5 if i % 2 == 0 else 100 - i * 0.5 for i in range(100)])
        
        # Запускаем на данных
        results = self.predictor.run_on_data(prices, verbose=False)
        
        # Проверяем, что результаты получены
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
    
    def test_with_ohlcv_data(self):
        """Тест работы с OHLCV данными"""
        # Создаем тестовые OHLCV данные
        ohlcv_data = []
        for i in range(100):
            price = 100 + i * 0.5
            ohlcv_data.append({
                'open': price - 0.2,
                'high': price + 0.3,
                'low': price - 0.3,
                'close': price,
                'volume': 10 + i * 0.1
            })
        
        # Создаем предиктор с использованием typical price
        config = PredictorConfig(
            window_size=50,
            prediction_depth=5,
            min_confidence=0.55,
            state_length=3,
            significant_change_pct=0.4,
            use_weighted_window=False,
            price_calculation='typical'
        )
        predictor = AdvancedMarkovPredictor(config)
        
        # Запускаем на данных
        results = predictor.run_on_data(ohlcv_data, verbose=False)
        
        # Проверяем, что результаты получены
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()