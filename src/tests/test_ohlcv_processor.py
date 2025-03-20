import unittest
import numpy as np
from src.data.ohlcv_processor import OHLCVProcessor

class TestOHLCVProcessor(unittest.TestCase):
    """Тесты для обработчика OHLCV данных"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.processor = OHLCVProcessor()
        self.symbol = "BTCUSDT"
        
        # Тестовая свеча в формате Binance
        self.test_kline = [
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
        
        # Тестовая свеча в формате словаря
        self.test_kline_dict = {
            'open_time': 1499040000000,
            'open': 8100.0,
            'high': 8150.0,
            'low': 8050.0,
            'close': 8120.0,
            'volume': 10.5,
            'close_time': 1499043600000,
            'quote_asset_volume': 85050.0,
            'number_of_trades': 120,
            'taker_buy_base_asset_volume': 5.5,
            'taker_buy_quote_asset_volume': 44680.0
        }
    
    def test_add_kline_list(self):
        """Тест добавления свечи в формате списка"""
        result = self.processor.add_kline(self.symbol, self.test_kline)
        self.assertTrue(result)
        
        # Проверяем, что данные добавлены
        df = self.processor.get_data_as_dataframe(self.symbol)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['close'].values[0], 8120.0)
    
    def test_add_kline_dict(self):
        """Тест добавления свечи в формате словаря"""
        result = self.processor.add_kline(self.symbol, self.test_kline_dict)
        self.assertTrue(result)
        
        # Проверяем, что данные добавлены
        df = self.processor.get_data_as_dataframe(self.symbol)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['close'].values[0], 8120.0)
    
    def test_add_klines(self):
        """Тест добавления нескольких свечей"""
        klines = []
        for i in range(5):
            # Копируем тестовую свечу и меняем время
            kline = self.test_kline.copy()
            kline[0] = kline[0] + i * 3600000  # Увеличиваем время на 1 час
            kline[6] = kline[6] + i * 3600000
            kline[4] = str(float(kline[4]) + i * 10)  # Меняем цену закрытия
            klines.append(kline)
        
        count = self.processor.add_klines(self.symbol, klines)
        self.assertEqual(count, 5)
        
        # Проверяем, что все данные добавлены
        df = self.processor.get_data_as_dataframe(self.symbol)
        self.assertEqual(len(df), 5)
    
    def test_get_prices(self):
        """Тест получения цен"""
        # Добавляем несколько свечей
        self.test_add_klines()
        
        # Получаем массив цен закрытия
        prices = self.processor.get_prices(self.symbol, 'close')
        self.assertEqual(len(prices), 5)
        self.assertTrue(np.all(prices > 8100.0))
    
    def test_calculate_price(self):
        """Тест расчета цены разными методами"""
        # Добавляем тестовую свечу
        self.processor.add_kline(self.symbol, self.test_kline)
        
        # Проверяем разные методы расчета
        single_price = self.processor.calculate_price(self.symbol, 'single', 'close')
        self.assertEqual(single_price[0], 8120.0)
        
        typical_price = self.processor.calculate_price(self.symbol, 'typical')
        self.assertEqual(typical_price[0], (8150.0 + 8050.0 + 8120.0) / 3)
        
        ohlc4_price = self.processor.calculate_price(self.symbol, 'ohlc4')
        self.assertEqual(ohlc4_price[0], (8100.0 + 8150.0 + 8050.0 + 8120.0) / 4)
    
    def test_clear_data(self):
        """Тест очистки данных"""
        # Добавляем данные
        self.processor.add_kline("BTCUSDT", self.test_kline)
        self.processor.add_kline("ETHUSDT", self.test_kline)
        
        # Очищаем данные по одному символу
        self.processor.clear_data("BTCUSDT")
        
        # Проверяем, что данные очищены
        df_btc = self.processor.get_data_as_dataframe("BTCUSDT")
        df_eth = self.processor.get_data_as_dataframe("ETHUSDT")
        self.assertEqual(len(df_btc), 0)
        self.assertEqual(len(df_eth), 1)
        
        # Очищаем все данные
        self.processor.clear_data()
        df_eth = self.processor.get_data_as_dataframe("ETHUSDT")
        self.assertEqual(len(df_eth), 0)

if __name__ == '__main__':
    unittest.main()