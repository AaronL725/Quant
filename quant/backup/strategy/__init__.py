"""
策略模块初始化文件
包含所有可用的交易策略
"""

from .base import StrategyBase
from .VWM_S_strategy import VWM_S
from .VWM_L_strategy import VWM_L
from .Swinger_S_strategy import Swinger_S
from .Swinger_L_strategy import Swinger_L
from .SupermanSystem_S_strategy import SupermanSystem_S
from .SupermanSystem_L_strategy import SupermanSystem_L
from .Reference_Deviation_System_S_strategy import Reference_Deviation_System_S
from .Reference_Deviation_System_L_strategy import Reference_Deviation_System_L
from .KingKeltner_S_strategy import KingKeltner_S
from .KingKeltner_L_strategy import KingKeltner_L
from .KeltnerChannel_S_strategy import KeltnerChannel_S
from .KeltnerChannel_L_strategy import KeltnerChannel_L
from .JailBreakSys_S_strategy import JailBreakSys_S
from .JailBreakSys_L_strategy import JailBreakSys_L
from .FourSetofMACrossoverSys_S_strategy import FourSetofMACrossoverSys_S
from .FourSetofMACrossoverSys_L_strategy import FourSetofMACrossoverSys_L
from .DynamicBreakOutII_S_strategy import DynamicBreakOutII_S
from .DynamicBreakOutII_L_strategy import DynamicBreakOutII_L
from .DisplacedBoll_S_strategy import DisplacedBoll_S
from .DisplacedBoll_L_strategy import DisplacedBoll_L
from .BollingerBandit_S_strategy import BollingerBandit_S
from .BollingerBandit_L_strategy import BollingerBandit_L
from .AverageChannelRangeLeader_S_strategy import AverageChannelRangeLeader_S
from .AverageChannelRangeLeader_L_strategy import AverageChannelRangeLeader_L
from .ADXandMAChannelSys_S_strategy import ADXandMAChannelSys_S
from .ADXandMAChannelSys_L_strategy import ADXandMAChannelSys_L
from .TrendScore_S_strategy import TrendScore_S
from .TrendScore_L_strategy import TrendScore_L
from .Trading_Range_Breakout_S_strategy import Trading_Range_Breakout_S
from .Trading_Range_Breakout_L_strategy import Trading_Range_Breakout_L
from .Three_EMA_Crossover_System_S_strategy import Three_EMA_Crossover_System_S
from .Three_EMA_Crossover_System_L_strategy import Three_EMA_Crossover_System_L
from .Thermostat_S_strategy import Thermostat_S
from .Thermostat_L_strategy import Thermostat_L
from .NoHurrySystem_S_strategy import NoHurrySystem_S
from .NoHurrySystem_L_strategy import NoHurrySystem_L
from .Going_in_Style_S_strategy import Going_in_Style_S
from .Going_in_Style_L_strategy import Going_in_Style_L
from .GhostTrader_S_strategy import GhostTrader_S
from .GhostTrader_L_strategy import GhostTrader_L
from .DualMA_strategy import DualMA
from .Traffic_Jam_S_strategy import Traffic_Jam_S
from .Traffic_Jam_L_strategy import Traffic_Jam_L
from .Open_Close_Histogram_S_strategy import Open_Close_Histogram_S
from .Open_Close_Histogram_L_strategy import Open_Close_Histogram_L
'''
#Short

##rugged
ADXandMAChannelSys_S_strategy
SupermanSystem_S_strategy
JailBreakSys_S_strategy
VWM_S_strategy

##down-up
Swinger_S_strategy
Reference_Deviation_System_S_strategy
KingKeltner_S_strategy
TrendScore_S_strategy
KeltnerChannel_S_strategy
FourSetofMACrossoverSys_S_strategy
DynamicBreakOutII_S_strategy
DisplacedBoll_S_strategy
BollingerBandit_S_strategy
AverageChannelRangeLeader_S_strategy
'''