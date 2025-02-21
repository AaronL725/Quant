'''
// 简称: BollingerBandit_L
// 名称: 布林强盗_多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        基于布林通道的突破系统
系统要素:
        1、基于收盘价计算而来的布林通道
        2、基于收盘价计算而来的进场过滤器
        3、自适应出场均线
入场条件:
        1、满足过滤条件，并且价格上破布林通道上轨，开多单
        2、满足过滤条件，并且价格下破布林通道下轨，开空单
出场条件:
        1、持有多单时，自适应出场均线低于布林通道上轨，并且价格下破自适应出场均线，平多单
        2、持有空单时，自适应出场均线高于布林通道下轨，并且价格上破自适应出场均线，平空单
注    意:
        此公式仅做多
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class BollingerBandit_L(StrategyBase):
    """布林强盗做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'bollingerLengths': 50,  # 布林通道参数
            'Offset': 1.25,          # 布林通道参数
            'rocCalcLength': 30,     # 过滤器参数
            'liqLength': 50,         # 自适应出场均线参数
            'Lots': 1                # 交易手数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 布林通道中轨
        df['MidLine'] = Average(df['close'], self.params['bollingerLengths'])
        
        # 计算标准差带
        df['Band'] = StandardDev(df['close'], self.params['bollingerLengths'], 2)
        
        # 布林通道上轨
        df['upBand'] = df['MidLine'] + self.params['Offset'] * df['Band']
        
        # 进场过滤器
        df['rocCalc'] = df['close'] - df['close'].shift(self.params['rocCalcLength'])
        
        # 初始化自适应出场均线参数
        df['liqDays'] = self.params['liqLength']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['bollingerLengths'], self.params['rocCalcLength'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态
        position_bars = 0     # 持仓天数
        
        for i in range(1, len(data)):
            if current_position == 0:  # 无持仓
                # 满足过滤条件，并且价格突破布林通道上轨，开多单
                if (data['rocCalc'].iloc[i-1] > 0 and 
                    data['high'].iloc[i] >= data['upBand'].iloc[i-1]):
                    signals.iloc[i] = [1]
                    current_position = 1
                    position_bars = 0
                    
            elif current_position == 1:  # 持有多仓
                position_bars += 1
                # 更新自适应出场均线参数
                data.loc[data.index[i], 'liqDays'] = max(data['liqDays'].iloc[i-1] - 1, 10)
                # 计算自适应出场均线
                liq_point = Average(data['close'].iloc[:i+1], int(data['liqDays'].iloc[i]))
                
                # 持有多单时，自适应出场均线低于布林通道上轨，并且价格下破自适应出场均线，平多单
                if (position_bars >= 1 and 
                    liq_point.iloc[-2] < data['upBand'].iloc[i-1] and 
                    data['low'].iloc[i] <= liq_point.iloc[-2]):
                    signals.iloc[i] = [0]
                    current_position = 0
                    
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = [0]
            
        return signals



#########################主函数#########################
def main():
    """主函数，执行多品种回测"""
    logger = setup_logging()
    
    level = 'day'  # 默认值
    valid_levels = {'min5', 'min15', 'min30', 'min60', 'day'}
    assert level in valid_levels, f"level必须是以下值之一: {valid_levels}"
    
    data_paths = {
        'open': rf'D:\pythonpro\python_test\quant\Data\{level}\open.csv',
        'close': rf'D:\pythonpro\python_test\quant\Data\{level}\close.csv',
        'high': rf'D:\pythonpro\python_test\quant\Data\{level}\high.csv',
        'low': rf'D:\pythonpro\python_test\quant\Data\{level}\low.csv',
        'vol': rf'D:\pythonpro\python_test\quant\Data\{level}\vol.csv'
    }
    
    try:
        data_cache = load_all_data(data_paths, logger, level)
        futures_codes = list(data_cache['open'].columns)
        
        config = {
            'data_paths': data_paths,
            'futures_codes': futures_codes,
            'start_date': '2023-02-28',
            'end_date': '2025-01-10',
            'initial_balance': 20000000.0
        }
        
        data_dict = load_data_vectorized(
            data_cache, 
            config['futures_codes'],
            config['start_date'],
            config['end_date'],
            logger
        )

        # 先计算所有品种的信号
        strategy = BollingerBandit_L()
        signals_dict = {}
        
        for code, data in data_dict.items():
            try:
                # 计算指标和信号
                data_with_indicators = strategy.calculate_indicators(data)
                signals = strategy.generate_signals(data_with_indicators)
                if isinstance(signals, pd.DataFrame) and len(signals) > 0:
                    signals_dict[code] = signals
            except Exception as e:
                logger.error(f"计算{code}信号时出错: {e}")
                continue
        
        # 将信号字典传入回测器
        backtester = Backtester(
            signals_dict=signals_dict,
            data_dict=data_dict,
            config=config,
            logger=logger,
            use_multiprocessing=True
        )
        t_pnl_df = backtester.run_backtest()
        
        plot_combined_pnl(t_pnl_df, logger)
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")


if __name__ == "__main__":
    main() 
