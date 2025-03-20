class PredictorConfig:
    """
    Класс для хранения и настройки параметров предиктора.
    """
    def __init__(self, 
                 window_size=550,            # Размер окна для анализа (количество точек)
                 prediction_depth=15,        # Глубина предсказания (на сколько точек вперед)
                 min_confidence=0.55,        # Минимальная уверенность для принятия решения
                 state_length=3,             # Длина паттерна состояния
                 significant_change_pct=0.4,  # Порог значимого изменения в процентах
                 use_weighted_window=False,   # Использовать взвешенное окно
                 weight_decay=0.95,          # Коэффициент затухания весов
                 adaptive_weighting=False,   # Использовать адаптивное взвешивание
                 volatility_weighting=False, # Использовать взвешивание по волатильности
                 recency_boost=1.5,          # Множитель увеличения веса недавних событий
                 focus_on_best_states=False, # Фокус только на лучших состояниях
                 best_states=None,           # Список лучших состояний
                 price_source="close",       # Какую цену использовать (open, high, low, close)
                 price_calculation="single", # Как рассчитывать цену (single, typical, hlc3)
                 volume_analysis=False):     # Использовать ли объем для анализа
        
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.min_confidence = min_confidence
        self.state_length = state_length
        self.significant_change_pct = significant_change_pct / 100  # Переводим проценты в доли
        self.use_weighted_window = use_weighted_window
        self.weight_decay = weight_decay
        self.adaptive_weighting = adaptive_weighting
        self.volatility_weighting = volatility_weighting
        self.recency_boost = recency_boost
        self.focus_on_best_states = focus_on_best_states
        self.best_states = best_states if best_states else []
        self.price_source = price_source
        self.price_calculation = price_calculation
        self.volume_analysis = volume_analysis
        
    def __str__(self):
        """Строковое представление конфигурации"""
        config_str = (f"PredictorConfig(window_size={self.window_size}, "
                     f"prediction_depth={self.prediction_depth}, "
                     f"min_confidence={self.min_confidence}, "
                     f"state_length={self.state_length}, "
                     f"significant_change_pct={self.significant_change_pct*100}%, "
                     f"use_weighted_window={self.use_weighted_window}, "
                     f"weight_decay={self.weight_decay}")
        
        if self.adaptive_weighting:
            config_str += f", adaptive_weighting=True"
        
        if self.volatility_weighting:
            config_str += f", volatility_weighting=True"
            
        if self.recency_boost != 1.0:
            config_str += f", recency_boost={self.recency_boost}"
        
        if self.focus_on_best_states:
            config_str += f", focus_on_best_states=True, best_states={self.best_states}"
            
        config_str += f", price_source={self.price_source}"
        config_str += f", price_calculation={self.price_calculation}"
        config_str += f", volume_analysis={self.volume_analysis}"
        
        config_str += ")"
        return config_str