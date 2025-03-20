import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class AdvancedMarkovPredictor:
    """
    Расширенный предиктор на основе марковского процесса со скользящим окном.
    Поддерживает настройку различных параметров для оптимизации.
    """
    
    def __init__(self, config=None):
        """
        Инициализация предиктора
        
        Параметры:
        config (PredictorConfig): конфигурация параметров предиктора
        """
        # Используем конфигурацию по умолчанию, если не передана
        self.config = config
        
        # Статистика
        self.total_predictions = 0
        self.correct_predictions = 0
        self.success_rate = 0.0
        
        # История статистики по точкам
        self.point_statistics = {}
        
        # Для детального анализа
        self.state_statistics = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    def _get_weights(self, window_length, prices=None, start_idx=None):
        """
        Генерирует веса для точек в окне с использованием различных схем взвешивания
        
        Параметры:
        window_length (int): длина окна
        prices (numpy.array, optional): массив цен для расчета волатильности
        start_idx (int, optional): начальный индекс в массиве цен
        
        Возвращает:
        numpy.array: массив весов
        """
        if not self.config.use_weighted_window:
            return np.ones(window_length)
        
        # Базовое экспоненциальное затухание - новые точки имеют больший вес
        weights = np.array([self.config.weight_decay ** i for i in range(window_length - 1, -1, -1)])
        
        # Адаптивное взвешивание - усиливаем значимость недавних точек
        if self.config.adaptive_weighting:
            # Определяем точку перегиба (середина окна)
            pivot = window_length // 2
            
            # Применяем усиленное взвешивание для последних точек
            for i in range(pivot, window_length):
                boost_factor = 1.0 + (i - pivot) / (window_length - pivot) * (self.config.recency_boost - 1.0)
                weights[i] *= boost_factor
        
        # Взвешивание по волатильности - точки с высокой волатильностью имеют меньший вес
        if self.config.volatility_weighting and prices is not None and start_idx is not None:
            volatility = []
            for i in range(window_length - 1):
                idx = start_idx + i
                if idx + 1 < len(prices):
                    # Вычисляем абсолютное процентное изменение
                    pct_change = abs((prices[idx + 1] - prices[idx]) / prices[idx]) if prices[idx] != 0 else 0
                    volatility.append(pct_change)
                else:
                    volatility.append(0)
            
            # Добавляем последнюю точку
            volatility.append(0)
            
            # Нормализуем волатильность
            max_vol = max(volatility) if max(volatility) > 0 else 1
            volatility = [v / max_vol for v in volatility]
            
            # Инвертируем веса волатильности (более стабильные периоды имеют больший вес)
            vol_weights = [1 - 0.5 * v for v in volatility]
            
            # Применяем веса волатильности
            weights = weights * np.array(vol_weights)
        
        # Нормализуем, чтобы сумма была равна window_length
        weights = weights * window_length / weights.sum()
        
        return weights
    
    def _determine_movement(self, current_price, next_price, current_data=None, next_data=None):
        """
        Определяет направление движения с учетом порога значимого изменения
        
        Parameters:
        current_price (float): текущая цена
        next_price (float): следующая цена
        current_data (dict, optional): дополнительные данные для текущей точки (high, low, volume и т.д.)
        next_data (dict, optional): дополнительные данные для следующей точки
        
        Returns:
        int: 1 = значимый рост, 2 = значимое падение, 0 = незначительное изменение
        """
        # Вычисляем процентное изменение
        if current_price == 0:
            return 0  # Избегаем деления на ноль
        
        pct_change = (next_price - current_price) / current_price
        
        # Если включен анализ объема, учитываем его при определении движения
        if self.config.volume_analysis and current_data and next_data:
            if 'volume' in current_data and 'volume' in next_data:
                current_volume = current_data['volume']
                next_volume = next_data['volume']
                
                # Если объем увеличился, это может усилить сигнал
                if next_volume > current_volume * 1.5:  # Объем вырос более чем на 50%
                    pct_change *= 1.2  # Усиливаем сигнал на 20%
        
        # Применяем порог значимого изменения
        if pct_change > self.config.significant_change_pct:
            return 1  # Значимый рост
        elif pct_change < -self.config.significant_change_pct:
            return 2  # Значимое падение
        else:
            return 0  # Незначительное изменение
        
    def _get_state(self, prices, idx, ohlcv_data=None):
        """
        Определяет текущее состояние рынка
        
        Parameters:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        ohlcv_data (list, optional): список словарей с OHLCV данными
        
        Returns:
        tuple: состояние рынка (последовательность движений)
        """
        # Нужно иметь как минимум state_length + 1 точек для определения состояния
        if idx < self.config.state_length:
            return None  # Недостаточно данных для определения состояния
        
        # Определяем последние state_length движений
        movements = []
        for i in range(idx - self.config.state_length, idx):
            # Если есть OHLCV данные, используем их
            if ohlcv_data:
                current_data = ohlcv_data[i] if i < len(ohlcv_data) else None
                next_data = ohlcv_data[i+1] if i+1 < len(ohlcv_data) else None
                movement = self._determine_movement(
                    prices[i], 
                    prices[i+1], 
                    current_data,
                    next_data
                )
            else:
                movement = self._determine_movement(prices[i], prices[i+1])
            
            # Для незначительных изменений (0) будем считать их как продолжение предыдущего движения
            # Если это первое движение в состоянии, считаем его нейтральным (1)
            if movement == 0:
                if len(movements) > 0:
                    movement = movements[-1]  # Продолжаем предыдущее движение
                else:
                    movement = 1  # По умолчанию нейтральное движение
            movements.append(movement)
        
        return tuple(movements)

    def _determine_outcome(self, prices, idx, ohlcv_data=None):
        """
        Определяет фактический исход через prediction_depth точек
        
        Parameters:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        ohlcv_data (list, optional): список словарей с OHLCV данными
        
        Returns:
        int: 1 = рост, 2 = падение, 0 = незначительное изменение
        """
        if idx + self.config.prediction_depth >= len(prices):
            return None  # Нет данных для проверки
        
        current_price = prices[idx]
        future_price = prices[idx + self.config.prediction_depth]
        
        # Если есть OHLCV данные, используем их
        if ohlcv_data:
            current_data = ohlcv_data[idx] if idx < len(ohlcv_data) else None
            future_data = ohlcv_data[idx + self.config.prediction_depth] if idx + self.config.prediction_depth < len(ohlcv_data) else None
            return self._determine_movement(
                current_price, 
                future_price, 
                current_data,
                future_data
            )
        else:
            # Используем тот же порог значимого изменения
            return self._determine_movement(current_price, future_price)
        
    def predict_at_point(self, prices, idx, ohlcv_data=None):
        """
        Делает предсказание в точке idx, анализируя предыдущие window_size точек
        
        Parameters:
        prices (numpy.array): массив цен
        idx (int): индекс точки для предсказания
        ohlcv_data (list, optional): список словарей с OHLCV данными
        
        Returns:
        dict: результат предсказания
        """
        if idx < self.config.window_size or idx < self.config.state_length:
            return {'prediction': 0, 'confidence': 0.0}  # Недостаточно данных
        
        # Определяем текущий паттерн движения с учетом OHLCV данных
        current_state = self._get_state(prices, idx, ohlcv_data)
        if current_state is None:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Если фокусируемся на лучших состояниях и текущее состояние не входит в список лучших,
        # то не делаем предсказания
        if self.config.focus_on_best_states and current_state not in self.config.best_states:
            return {'prediction': 0, 'confidence': 0.0, 'state': current_state}
        
        # Определяем окно для анализа
        start_idx = max(self.config.state_length, idx - self.config.window_size)
        window = prices[start_idx:idx+1]
        
        # Получаем веса для точек в окне с передачей массива цен и начального индекса
        weights = self._get_weights(len(window) - self.config.state_length, prices, start_idx)
        
        # Собираем статистику переходов в этом окне
        transitions = defaultdict(lambda: {1: 0, 2: 0})
        weighted_transitions = defaultdict(lambda: {1: 0, 2: 0})
        
        # Проходим по окну, собирая статистику
        valid_points = 0
        for i in range(self.config.state_length, len(window) - self.config.prediction_depth):
            # Вычисляем абсолютный индекс
            abs_idx = start_idx + i
            
            # Определяем состояние в этой точке с учетом OHLCV данных
            state = self._get_state(prices, abs_idx, ohlcv_data)
            if state is None:
                continue
                
            # Если фокусируемся на лучших состояниях и текущее состояние не входит в список лучших,
            # пропускаем его при сборе статистики
            if self.config.focus_on_best_states and state not in self.config.best_states:
                continue
            
            # Определяем, что произошло через prediction_depth точек с учетом OHLCV данных
            outcome = self._determine_outcome(prices, abs_idx, ohlcv_data)
            if outcome is None or outcome == 0:  # Пропускаем незначительные изменения
                continue
            
            # Вес для этой точки
            weight = weights[valid_points] if self.config.use_weighted_window else 1.0
            valid_points += 1
            
            # Обновляем счетчики переходов
            transitions[state][outcome] += 1
            weighted_transitions[state][outcome] += weight
        
        # Если для текущего состояния нет статистики, не делаем предсказаний
        if current_state not in transitions:
            return {'prediction': 0, 'confidence': 0.0, 'state': current_state}
        
        # Используем взвешенные переходы, если включено взвешивание
        transition_counts = weighted_transitions[current_state] if self.config.use_weighted_window else transitions[current_state]
        
        # Вычисляем вероятности переходов для текущего состояния
        total = sum(transition_counts.values())
        probabilities = {}
        for outcome, count in transition_counts.items():
            if total > 0:
                probabilities[outcome] = count / total
        
        # Получаем вероятности
        up_prob = probabilities.get(1, 0)
        down_prob = probabilities.get(2, 0)
        
        # Выбираем предсказание с наибольшей вероятностью, если оно превышает порог
        if up_prob > down_prob and up_prob >= self.config.min_confidence:
            prediction = 1
            confidence = up_prob
        elif down_prob > up_prob and down_prob >= self.config.min_confidence:
            prediction = 2
            confidence = down_prob
        else:
            # Если нет явного преимущества или уверенность ниже порога
            prediction = 0
            confidence = max(up_prob, down_prob)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'up_prob': up_prob,
            'down_prob': down_prob,
            'state': current_state,
            'state_occurrences': total
        }
    
    def run_on_data(self, data, verbose=True):
        """
        Последовательно проходит по данным, делая предсказания и проверяя их
        
        Parameters:
        data (list/numpy.array/pandas.DataFrame): данные для анализа
            - Если numpy.array: предполагается, что это массив цен
            - Если list of dict: предполагается, что это список OHLCV данных
            - Если pandas.DataFrame: предполагается, что это DataFrame с OHLCV данными
        verbose (bool): выводить информацию о прогрессе
        
        Returns:
        list: результаты предсказаний
        """
        results = []
        
        # Преобразуем данные в нужный формат
        prices = []
        ohlcv_data = []
        
        if isinstance(data, np.ndarray):
            # Если данные - это numpy array, предполагаем, что это просто массив цен
            prices = data
            ohlcv_data = [{'close': p} for p in data]
        
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Если данные - это список словарей, предполагаем, что это OHLCV данные
            prices = np.array([self._calculate_price(item) for item in data])
            ohlcv_data = data
        
        elif isinstance(data, pd.DataFrame):
            # Если данные - это pandas DataFrame, предполагаем, что это OHLCV данные
            # Преобразуем DataFrame в список словарей
            ohlcv_data = data.to_dict('records')
            prices = np.array([self._calculate_price(item) for item in ohlcv_data])
        
        else:
            raise ValueError("Unsupported data format. Expected numpy.array, list of dict, or pandas.DataFrame")
        
        # Начинаем с точки, где у нас достаточно данных для анализа
        min_idx = max(self.config.window_size, self.config.state_length)
        
        for idx in tqdm(range(min_idx, len(prices) - self.config.prediction_depth), disable=not verbose):
            # Делаем предсказание, анализируя предыдущие window_size точек
            pred_result = self.predict_at_point(prices, idx, ohlcv_data)
            prediction = pred_result['prediction']
            
            # Если предсказание не "не знаю", проверяем результат через prediction_depth точек
            if prediction != 0:
                actual_outcome = self._determine_outcome(prices, idx, ohlcv_data)
                # Пропускаем проверку, если результат незначительное изменение (0)
                if actual_outcome is None or actual_outcome == 0:
                    continue
                
                is_correct = (prediction == actual_outcome)
                
                # Обновляем статистику
                self.total_predictions += 1
                if is_correct:
                    self.correct_predictions += 1
                
                # Обновляем успешность
                self.success_rate = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
                
                # Сохраняем статистику для этой точки
                self.point_statistics[idx] = {
                    'correct': self.correct_predictions,
                    'total': self.total_predictions,
                    'success_rate': self.success_rate
                }
                
                # Обновляем статистику по этому состоянию
                state = pred_result['state']
                self.state_statistics[state]['total'] += 1
                if is_correct:
                    self.state_statistics[state]['correct'] += 1
                
                # Сохраняем результат
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': prediction,
                    'actual': actual_outcome,
                    'is_correct': is_correct,
                    'confidence': pred_result['confidence'],
                    'success_rate': self.success_rate,
                    'correct_total': f"{self.correct_predictions}-{self.total_predictions}",
                    'state': pred_result['state'],
                    'state_occurrences': pred_result.get('state_occurrences', 0)
                }
            else:
                # Если предсказание "не знаю"
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': 0,
                    'confidence': pred_result.get('confidence', 0.0),
                    'success_rate': self.success_rate if self.total_predictions > 0 else 0,
                    'state': pred_result.get('state'),
                    'state_occurrences': pred_result.get('state_occurrences', 0)
                }
            
            results.append(result)
            
            # Выводим промежуточные результаты
            if verbose and idx % 1000 == 0 and self.total_predictions > 0:
                print(f"Точка {idx}: Успешность {self.success_rate*100:.2f}% ({self.correct_predictions}/{self.total_predictions})")
        
        return results
    
    def visualize_results(self, prices, results, save_path=None):
        """
        Визуализирует результаты предсказаний
        
        Параметры:
        prices (numpy.array): массив цен
        results (list): результаты предсказаний
        save_path (str): путь для сохранения графиков
        """
        # Создаем графики
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # График цен
        ax1.plot(prices, color='blue', alpha=0.7, label='Цена')
        
        # Выделяем обучающий участок
        ax1.axvspan(0, self.config.window_size, color='lightgray', alpha=0.3, label='Начальное окно')
        
        # Отмечаем предсказания
        correct_up_indices = []
        correct_up_prices = []
        correct_down_indices = []
        correct_down_prices = []
        
        wrong_up_indices = []
        wrong_up_prices = []
        wrong_down_indices = []
        wrong_down_prices = []
        
        for r in results:
            idx = r['index']
            price = r['price']
            
            if 'is_correct' in r:
                if r['prediction'] == 1:  # Up
                    if r['is_correct']:
                        correct_up_indices.append(idx)
                        correct_up_prices.append(price)
                    else:
                        wrong_up_indices.append(idx)
                        wrong_up_prices.append(price)
                elif r['prediction'] == 2:  # Down
                    if r['is_correct']:
                        correct_down_indices.append(idx)
                        correct_down_prices.append(price)
                    else:
                        wrong_down_indices.append(idx)
                        wrong_down_prices.append(price)
        
        # Отмечаем предсказания на графике
        if correct_up_indices:
            ax1.scatter(correct_up_indices, correct_up_prices, color='green', marker='^', s=50, alpha=0.7, 
                      label='Верно (Рост)')
        if correct_down_indices:
            ax1.scatter(correct_down_indices, correct_down_prices, color='green', marker='v', s=50, alpha=0.7, 
                      label='Верно (Падение)')
        if wrong_up_indices:
            ax1.scatter(wrong_up_indices, wrong_up_prices, color='red', marker='^', s=50, alpha=0.7, 
                      label='Неверно (Рост)')
        if wrong_down_indices:
            ax1.scatter(wrong_down_indices, wrong_down_prices, color='red', marker='v', s=50, alpha=0.7, 
                      label='Неверно (Падение)')
        
        ax1.set_title('Цена и предсказания')
        ax1.set_ylabel('Цена')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # График успешности предсказаний
        success_indices = []
        success_rates = []
        
        for idx, stats in sorted(self.point_statistics.items()):
            success_indices.append(idx)
            success_rates.append(stats['success_rate'] * 100)
        
        if success_indices:
            ax2.plot(success_indices, success_rates, 'g-', linewidth=2)
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Динамика успешности предсказаний')
            ax2.set_xlabel('Индекс')
            ax2.set_ylabel('Успешность (%)')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def get_state_statistics(self):
        """
        Возвращает статистику по состояниям
        
        Возвращает:
        pandas.DataFrame: статистика по состояниям
        """
        data = []
        for state, stats in self.state_statistics.items():
            success_rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            data.append({
                'state': str(state),
                'total': stats['total'],
                'correct': stats['correct'],
                'success_rate': success_rate
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('total', ascending=False)
    
    def _calculate_price(self, data):
        """
        Рассчитывает цену в зависимости от настроек конфигурации
        
        Parameters:
        data (dict): данные свечи (OHLCV)
        
        Returns:
        float: рассчитанная цена
        """
        if not data:
            return 0.0
        
        # Получаем метод расчета цены из конфигурации
        price_calculation = getattr(self.config, 'price_calculation', 'single')
        price_source = getattr(self.config, 'price_source', 'close')
        
        if price_calculation == 'single':
            # Используем один источник цены
            return float(data.get(price_source, data.get('close', 0.0)))
        
        elif price_calculation == 'typical' or price_calculation == 'hlc3':
            # Используем типичную цену (high + low + close) / 3
            high = float(data.get('high', 0.0))
            low = float(data.get('low', 0.0))
            close = float(data.get('close', 0.0))
            return (high + low + close) / 3
        
        elif price_calculation == 'ohlc4':
            # Средняя цена (open + high + low + close) / 4
            open_price = float(data.get('open', 0.0))
            high = float(data.get('high', 0.0))
            low = float(data.get('low', 0.0))
            close = float(data.get('close', 0.0))
            return (open_price + high + low + close) / 4
        
        else:
            # По умолчанию используем close
            return float(data.get('close', 0.0))
    
    def generate_report(self, results, save_path=None):
        """
        Генерирует подробный отчет о результатах предсказаний
        
        Параметры:
        results (list): результаты предсказаний
        save_path (str): путь для сохранения отчета
        
        Возвращает:
        str: текст отчета
        """
        # Общая статистика
        total_predictions = self.total_predictions
        correct_predictions = self.correct_predictions
        success_rate = self.success_rate * 100
        
        # Распределение предсказаний
        up_count = sum(1 for r in results if r.get('prediction') == 1)
        down_count = sum(1 for r in results if r.get('prediction') == 2)
        neutral_count = sum(1 for r in results if r.get('prediction') == 0)
        
        # Успешность по типам предсказаний
        up_correct = sum(1 for r in results if r.get('prediction') == 1 and r.get('is_correct', False))
        down_correct = sum(1 for r in results if r.get('prediction') == 2 and r.get('is_correct', False))
        
        up_success_rate = up_correct / up_count * 100 if up_count > 0 else 0
        down_success_rate = down_correct / down_count * 100 if down_count > 0 else 0
        
        # Топ состояния
        state_stats = self.get_state_statistics()
        top_states = state_stats.head(10)
        
        # Формируем отчет
        report = f"""
# Отчет о работе предиктора

## Конфигурация
{str(self.config)}

## Общая статистика
- Всего предсказаний: {total_predictions}
- Правильных предсказаний: {correct_predictions}
- Успешность: {success_rate:.2f}%

## Распределение предсказаний
- Рост: {up_count} ({up_count/len(results)*100:.2f}%)
- Падение: {down_count} ({down_count/len(results)*100:.2f}%)
- Не знаю: {neutral_count} ({neutral_count/len(results)*100:.2f}%)

## Успешность по типам предсказаний
- Успешность предсказаний роста: {up_correct}/{up_count} ({up_success_rate:.2f}%)
- Успешность предсказаний падения: {down_correct}/{down_count} ({down_success_rate:.2f}%)

## Топ-10 состояний по частоте
{top_states.to_markdown(index=False)}

## Покрытие предсказаний
- Общее покрытие: {(up_count + down_count) / len(results) * 100:.2f}%
"""
        
        # Сохраняем отчет, если указан путь
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def set_filters(self, filters_dict):
        """
        Устанавливает фильтры для предиктора
        
        Параметры:
        filters_dict (dict): словарь с фильтрами и их настройками
        """
        # Этот метод можно реализовать позже
        # Он будет устанавливать фильтры для обработки данных
        pass