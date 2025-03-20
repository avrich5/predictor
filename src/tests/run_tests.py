import unittest
import sys
import os

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортируем тесты
from src.tests.test_api import TestAPI
from src.tests.test_predictor import TestPredictor
from src.tests.test_ohlcv_processor import TestOHLCVProcessor
from src.tests.test_integration import TestIntegration

if __name__ == '__main__':
    # Создаем и запускаем тесты
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPredictor))
    suite.addTest(unittest.makeSuite(TestOHLCVProcessor))
    suite.addTest(unittest.makeSuite(TestIntegration))
    suite.addTest(unittest.makeSuite(TestAPI))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выходим с кодом ошибки, если тесты не прошли
    sys.exit(not result.wasSuccessful())