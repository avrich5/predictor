import requests
import time
from datetime import datetime

class BinanceClient:
    """
    Клиент для работы с Binance API
    """
    
    def __init__(self, base_url="https://api.binance.com"):
        self.base_url = base_url
    
    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        Получает исторические Kline/Candlestick данные с Binance
        
        Parameters:
        symbol (str): Символ (пара), например, 'BTCUSDT'
        interval (str): Интервал свечи, например, '1m', '5m', '1h', '1d'
        start_time (int, optional): Начальное время в миллисекундах
        end_time (int, optional): Конечное время в миллисекундах
        limit (int): Количество свечей (максимум 1000)
        
        Returns:
        list: Массив свечей в формате Binance
        """
        try:
            endpoint = "/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)  # Binance ограничивает до 1000
            }
            
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting klines: {response.status_code} {response.text}")
                return []
        
        except Exception as e:
            print(f"Error getting klines: {e}")
            return []
    
    def get_all_klines(self, symbol, interval, start_time=None, end_time=None, max_requests=10):
        """
        Получает все исторические данные за указанный период,
        делая несколько запросов, если нужно
        
        Parameters:
        symbol (str): Символ (пара), например, 'BTCUSDT'
        interval (str): Интервал свечи, например, '1m', '5m', '1h', '1d'
        start_time (int, optional): Начальное время в миллисекундах
        end_time (int, optional): Конечное время в миллисекундах
        max_requests (int): Максимальное количество запросов
        
        Returns:
        list: Массив свечей в формате Binance
        """
        all_klines = []
        current_start = start_time
        
        for _ in range(max_requests):
            # Получаем порцию данных
            klines = self.get_klines(symbol, interval, current_start, end_time, 1000)
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Если получили меньше 1000 свечей, значит достигли конца
            if len(klines) < 1000:
                break
            
            # Обновляем время начала для следующего запроса
            # +1 к времени закрытия последней свечи
            current_start = klines[-1][6] + 1
            
            # Делаем паузу между запросами, чтобы не превысить лимиты
            time.sleep(0.5)
        
        return all_klines
    
    def get_interval_milliseconds(self, interval):
        """
        Преобразует строковый интервал в миллисекунды
        
        Parameters:
        interval (str): Интервал, например, '1m', '5m', '1h', '1d'
        
        Returns:
        int: Интервал в миллисекундах
        """
        # Словарь для преобразования интервалов
        intervals = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000
        }
        
        return intervals.get(interval, 0)