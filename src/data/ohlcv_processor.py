import numpy as np
import pandas as pd
from datetime import datetime

class OHLCVProcessor:
    """
    Класс для обработки и хранения OHLCV (Open, High, Low, Close, Volume) данных
    """
    
    def __init__(self):
        self.data = pd.DataFrame()
        self.symbols = {}  # Словарь для хранения данных по разным символам
    
    def add_kline(self, symbol, kline, replace_existing=False):
        """
        Добавляет одну свечу (kline) в хранилище данных
        
        Parameters:
        symbol (str): Символ (пара) для которой добавляется свеча
        kline (list/dict): Данные свечи в формате Binance или словаря
        replace_existing (bool): Заменить существующую свечу, если она уже есть
        
        Returns:
        bool: True если операция успешна
        """
        try:
            # Преобразуем kline в словарь, если он пришел в формате списка Binance
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
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_asset_volume': float(kline[9]),
                    'taker_buy_quote_asset_volume': float(kline[10])
                }
            else:
                kline_dict = kline
            
            # Создаем DataFrame для этой свечи
            df = pd.DataFrame([kline_dict])
            
            # Если данных для этого символа еще нет, создаем новый DataFrame
            if symbol not in self.symbols:
                self.symbols[symbol] = df
            else:
                # Проверяем, есть ли уже такая свеча (по open_time)
                existing = self.symbols[symbol]
                if 'open_time' in existing.columns and 'open_time' in df.columns:
                    # Если свеча с таким временем уже существует и replace_existing=True, заменяем её
                    if replace_existing:
                        self.symbols[symbol] = pd.concat([
                            existing[existing['open_time'] != df['open_time'].values[0]], 
                            df
                        ]).reset_index(drop=True)
                    else:
                        # Иначе добавляем только если такой свечи еще нет
                        if df['open_time'].values[0] not in existing['open_time'].values:
                            self.symbols[symbol] = pd.concat([existing, df]).reset_index(drop=True)
                else:
                    # Если нет колонки open_time, просто добавляем
                    self.symbols[symbol] = pd.concat([existing, df]).reset_index(drop=True)
            
            return True
            
        except Exception as e:
            print(f"Error adding kline: {e}")
            return False
    
    def add_klines(self, symbol, klines):
        """
        Добавляет массив свечей в хранилище данных
        
        Parameters:
        symbol (str): Символ (пара) для которой добавляются свечи
        klines (list): Массив свечей в формате Binance
        
        Returns:
        int: Количество добавленных свечей
        """
        count = 0
        for kline in klines:
            if self.add_kline(symbol, kline):
                count += 1
        return count
    
    def get_prices(self, symbol, price_source='close', start_time=None, end_time=None):
        """
        Получает массив цен для указанного символа
        
        Parameters:
        symbol (str): Символ (пара) для которой нужны цены
        price_source (str): Тип цены (open, high, low, close)
        start_time (int): Начальное время в миллисекундах
        end_time (int): Конечное время в миллисекундах
        
        Returns:
        numpy.array: Массив цен
        """
        if symbol not in self.symbols:
            return np.array([])
        
        df = self.symbols[symbol]
        
        # Фильтрация по времени, если указано
        if start_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] >= start_time]
        
        if end_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] <= end_time]
        
        # Получаем нужный столбец цены
        if price_source in df.columns:
            return df[price_source].values
        else:
            return np.array([])
    
    def calculate_price(self, symbol, method='single', price_source='close', start_time=None, end_time=None):
        """
        Рассчитывает цену по указанному методу
        
        Parameters:
        symbol (str): Символ (пара) для которой нужны цены
        method (str): Метод расчета цены:
            - single: использовать указанный price_source
            - typical: (high + low + close) / 3
            - hlc3: то же, что и typical
            - ohlc4: (open + high + low + close) / 4
        price_source (str): Тип цены для метода 'single'
        start_time (int): Начальное время в миллисекундах
        end_time (int): Конечное время в миллисекундах
        
        Returns:
        numpy.array: Массив рассчитанных цен
        """
        if symbol not in self.symbols:
            return np.array([])
        
        df = self.symbols[symbol]
        
        # Фильтрация по времени, если указано
        if start_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] >= start_time]
        
        if end_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] <= end_time]
        
        # Расчет цены по указанному методу
        if method == 'single':
            if price_source in df.columns:
                return df[price_source].values
            else:
                return np.array([])
        elif method in ['typical', 'hlc3']:
            if all(col in df.columns for col in ['high', 'low', 'close']):
                return ((df['high'] + df['low'] + df['close']) / 3).values
            else:
                return np.array([])
        elif method == 'ohlc4':
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return ((df['open'] + df['high'] + df['low'] + df['close']) / 4).values
            else:
                return np.array([])
        else:
            # Неизвестный метод, возвращаем пустой массив
            return np.array([])
    
    def get_data_as_dataframe(self, symbol, start_time=None, end_time=None):
        """
        Получает данные в виде pandas DataFrame
        
        Parameters:
        symbol (str): Символ (пара) для которой нужны данные
        start_time (int): Начальное время в миллисекундах
        end_time (int): Конечное время в миллисекундах
        
        Returns:
        pandas.DataFrame: DataFrame с данными
        """
        if symbol not in self.symbols:
            return pd.DataFrame()
        
        df = self.symbols[symbol]
        
        # Фильтрация по времени, если указано
        if start_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] >= start_time]
        
        if end_time is not None and 'open_time' in df.columns:
            df = df[df['open_time'] <= end_time]
        
        return df.copy()
    
    def clear_data(self, symbol=None):
        """
        Очищает хранилище данных
        
        Parameters:
        symbol (str, optional): Символ для очистки. Если None, очищаются все данные
        """
        if symbol is None:
            self.symbols = {}
        elif symbol in self.symbols:
            del self.symbols[symbol]