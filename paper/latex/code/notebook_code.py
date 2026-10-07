# ---- Cell 3 (run 1) ----
!pip install pandas-ta
!pip install stable-baselines3
!pip install sklearn

# ---- Cell 4 (run 2) ----
import aif_environment
import yfinance as yf
import pandas as pd
import numpy as np

# ---- Cell 6 (run 3) ----
boeing = yf.Ticker('BA')
ba_df = boeing.history(period="10y")
ba_df['Close'].plot(title="BA's stock price")

# ---- Cell 8 (run 4) ----
airbus = yf.Ticker('AIR.PA')
air_df = airbus.history(period="10y")
air_df['Close'].plot(title="AIR's stock price")

# ---- Cell 10 (run 5) ----
toyota = yf.Ticker('TM')
tm_df = toyota.history(period="10y")
tm_df['Close'].plot(title="TM's stock price")

# ---- Cell 13 (run 6) ----
print((boeing.info.get('trailingPE')))

# ---- Cell 15 (run 7) ----
print((airbus.info.get('trailingPE')))

# ---- Cell 17 (run 8) ----
print((toyota.info.get('trailingPE')))

# ---- Cell 20 (run 9) ----
print((boeing.info['priceToSalesTrailing12Months']))

# ---- Cell 22 (run 10) ----
print((airbus.info['priceToSalesTrailing12Months']))

# ---- Cell 24 (run 11) ----
print((toyota.info['priceToSalesTrailing12Months']))

# ---- Cell 28 (run 12) ----
bo = yf.download('BA', start = '2021-09-01', end='2022-08-31')
bo.reset_index(inplace=True)

# ---- Cell 29 (run 13) ----
bo['Log_Returns'] = np.log(bo['Adj Close']) - np.log(bo['Adj Close'].shift(1))
bo.head()

# ---- Cell 30 (run 14) ----
bo['Log_Returns'].sum()

# ---- Cell 32 (run 15) ----
ai = yf.download('AIR.PA', start = '2021-09-01', end='2022-08-31')
ai.reset_index(inplace=True)

# ---- Cell 33 (run 16) ----
ai['Log_Returns'] = np.log(ai['Adj Close']) - np.log(ai['Adj Close'].shift(1))
ai.head()

# ---- Cell 34 (run 17) ----
ai['Log_Returns'].sum()

# ---- Cell 36 (run 18) ----
tm = yf.download('TM', start = '2021-09-01', end='2022-08-31')
tm.reset_index(inplace=True)

# ---- Cell 37 (run 19) ----
tm['Log_Returns'] = np.log(tm['Adj Close']) - np.log(tm['Adj Close'].shift(1))
tm.head()

# ---- Cell 38 (run 20) ----
tm['Log_Returns'].sum()

# ---- Cell 41 (run 21) ----
from stable_baselines3 import A2C

# load the trained agents
model5 = A2C.load('aif_course-master/A2C_showcase_agent5')
model4 = A2C.load('aif_course-master/A2C_showcase_agent4')
model3 = A2C.load('aif_course-master/A2C_showcase_agent3')
model2 = A2C.load('aif_course-master/A2C_showcase_agent2')
model = A2C.load('aif_course-master/A2C_showcase_agent1')

# ---- Cell 42 (run 22) ----
from aif_environment import TradingEnvironment
from aif_analysis import ai_trade_performance

# ---- Cell 43 (run 23) ----
env = TradingEnvironment(ticker = 'BA', start_date = '2012-01-01', end_date = '2017-12-31')

# and examine the agent's performance
res = ai_trade_performance(env, model5, num_plays = 20)
res

# ---- Cell 44 (run 24) ----
res = ai_trade_performance(env, model4, num_plays = 20)
res

# ---- Cell 45 (run 25) ----
res = ai_trade_performance(env, model3, num_plays = 20)
res

# ---- Cell 46 (run 26) ----
res = ai_trade_performance(env, model2, num_plays = 20)
res

# ---- Cell 47 (run 27) ----
res = ai_trade_performance(env, model, num_plays = 20)
res

# ---- Cell 48 (run 28) ----
from aif_analysis import analyze_actions_taken

analyze_actions_taken(env, model, plot_actions = True)

# ---- Cell 49 (run 29) ----
analyze_actions_taken(env, model2, plot_actions = True)

# ---- Cell 50 (run 30) ----
analyze_actions_taken(env, model3, plot_actions = True)

# ---- Cell 51 (run 31) ----
analyze_actions_taken(env, model4, plot_actions = True)

# ---- Cell 52 (run 32) ----
analyze_actions_taken(env, model5, plot_actions = True)

# ---- Cell 54 (run 33) ----
returns_ta = np.array ([[1.151250], [0.982355], [1.268870], [1.211902], [1.220214]])
returns_bh = np.array([[1.248442], [1.194680], [1.411054], [1.325877], [1.139906]])
agent_better = ([0.55], [0.1], [0.4], [0.8], [0.4])

average_return_ta = np.mean(returns_ta)
print("The average return of all trained agents is:", average_return_ta)
average_return_bh = np.mean(returns_bh)
print("The average return of the market over all 5 iterations is:",average_return_bh)
agent_better_average = np.mean(agent_better)
print("The average in which the agent performs better than the market:",agent_better_average)

# ---- Cell 57 (run 34) ----
model5_AIR = A2C.load('aif_course-master/A2C_showcase_agent5_AIR')
model4_AIR = A2C.load('aif_course-master/A2C_showcase_agent4_AIR')
model3_AIR = A2C.load('aif_course-master/A2C_showcase_agent3_AIR')
model2_AIR = A2C.load('aif_course-master/A2C_showcase_agent2_AIR')
model_AIR = A2C.load('aif_course-master/A2C_showcase_agent1_AIR')

# ---- Cell 58 (run 35) ----
env_AIR = TradingEnvironment(ticker = 'AIR.PA', start_date = '2012-01-01', end_date = '2017-12-31')

# and examine the agent's performance
res = ai_trade_performance(env_AIR, model_AIR, num_plays = 20)
res

# ---- Cell 59 (run 36) ----
res = ai_trade_performance(env_AIR, model2_AIR, num_plays = 20)
res

# ---- Cell 60 (run 37) ----
res = ai_trade_performance(env_AIR, model3_AIR, num_plays = 20)
res

# ---- Cell 61 (run 38) ----
res = ai_trade_performance(env_AIR, model4_AIR, num_plays = 20)
res

# ---- Cell 62 (run 39) ----
res = ai_trade_performance(env_AIR, model5_AIR, num_plays = 20)
res

# ---- Cell 63 (run 40) ----
from aif_analysis import analyze_actions_taken

analyze_actions_taken(env_AIR, model_AIR, plot_actions = True)

# ---- Cell 64 (run 41) ----
analyze_actions_taken(env_AIR, model2_AIR, plot_actions = True)

# ---- Cell 65 (run 42) ----
analyze_actions_taken(env_AIR, model3_AIR, plot_actions = True)

# ---- Cell 66 (run 43) ----
analyze_actions_taken(env_AIR, model4_AIR, plot_actions = True)

# ---- Cell 67 (run 44) ----
analyze_actions_taken(env_AIR, model5_AIR, plot_actions = True)

# ---- Cell 69 (run 45) ----
returns_ta = np.array ([[1.430267], [1.570172], [2.007212], [1.276262], [1.038152]])
returns_bh = np.array([[1.217388], [1.188968], [1.168107], [1.195885], [1.237382]])
agent_better = ([0.75], [0.60], [0.80], [0.60], [0.20])

average_return_ta = np.mean(returns_ta)
print("The average return of all trained agents is:", average_return_ta)
average_return_bh = np.mean(returns_bh)
print("The average return of the market over all 5 iterations is:",average_return_bh)
agent_better_average = np.mean(agent_better)
print("The average in which the agent performs better than the market:",agent_better_average)

# ---- Cell 72 (run 46) ----

# load the trained agents
model5_TM = A2C.load('aif_course-master/A2C_showcase_agent5_TM')
model4_TM = A2C.load('aif_course-master/A2C_showcase_agent4_TM')
model3_TM = A2C.load('aif_course-master/A2C_showcase_agent3_TM')
model2_TM = A2C.load('aif_course-master/A2C_showcase_agent2_TM')
model_TM = A2C.load('aif_course-master/A2C_showcase_agent1_TM')

# ---- Cell 73 (run 47) ----
env_TM = TradingEnvironment(ticker = 'TM', start_date = '2012-01-01', end_date = '2017-12-31')

# and examine the agent's performance
res = ai_trade_performance(env_TM, model_TM, num_plays = 20)
res

# ---- Cell 74 (run 48) ----
res = ai_trade_performance(env_TM, model2_TM, num_plays = 20)
res

# ---- Cell 75 (run 49) ----
res = ai_trade_performance(env_TM, model3_TM, num_plays = 20)
res

# ---- Cell 76 (run 50) ----
res = ai_trade_performance(env_TM, model4_TM, num_plays = 20)
res

# ---- Cell 77 (run 51) ----
res = ai_trade_performance(env_TM, model5_TM, num_plays = 20)
res

# ---- Cell 78 (run 52) ----
analyze_actions_taken(env_TM, model_TM, plot_actions = True)

# ---- Cell 79 (run 53) ----
analyze_actions_taken(env_TM, model2_TM, plot_actions = True)

# ---- Cell 80 (run 54) ----
analyze_actions_taken(env_TM, model3_TM, plot_actions = True)

# ---- Cell 81 (run 55) ----
analyze_actions_taken(env_TM, model4_TM, plot_actions = True)

# ---- Cell 82 (run 56) ----
analyze_actions_taken(env_TM, model5_TM, plot_actions = True)

# ---- Cell 84 (run 57) ----
returns_ta = np.array ([[1.160707], [1.442139], [1.185587], [1.263972], [1.272148]])
returns_bh = np.array([[0.994851], [1.014057], [1.044513], [1.07005], [1.003871]])
agent_better = ([0.80], [0.70], [0.80], [0.85], [0.75])

average_return_ta = np.mean(returns_ta)
print("The average return of all trained agents is:", average_return_ta)
average_return_bh = np.mean(returns_bh)
print("The average return of the market over all 5 iterations is:",average_return_bh)
agent_better_average = np.mean(agent_better)
print("The average in which the agent performs better than the market:",agent_better_average)

# ---- Cell 100 (run 58) ----
import yfinance as yf
import pandas as pd

stock = yf.Ticker("TM")

df = stock.history(start = "2012-01-01", end = "2017-12-31")
df.ta.strategy()
df["STOCHd_14_3_3"]

# ---- Cell 101 (run 59) ----
from aif_environment import TradingEnvironment

my_indicators = ["AROONOSC_14", "RSI_14", "MACD_12_26_9", "OBV", "STOCHk_14_3_3", "STOCHd_14_3_3"]

env_ti = TradingEnvironment(ticker = "TM", start_date = '2012-01-01', end_date = '2017-12-31', use_variables = my_indicators)

# ---- Cell 102 (run 60) ----
from stable_baselines3 import A2C

# ---- Cell 103 (run 61) ----
model_ti5 = A2C('MlpPolicy', env_ti, verbose = 1)
model_ti5.learn(total_timesteps = 75000)

# ---- Cell 104 (run 62) ----
model_ti4 = A2C('MlpPolicy', env_ti, verbose = 0)
model_ti4.learn(total_timesteps = 75000)

# ---- Cell 105 (run 63) ----
model_ti3 = A2C('MlpPolicy', env_ti, verbose = 0)
model_ti3.learn(total_timesteps = 75000)

# ---- Cell 106 (run 64) ----
model_ti2 = A2C('MlpPolicy', env_ti, verbose = 0)
model_ti2.learn(total_timesteps = 75000)

# ---- Cell 107 (run 65) ----
model_ti = A2C('MlpPolicy', env_ti, verbose = 0)
model_ti.learn(total_timesteps = 75000)

# ---- Cell 108 (run 66) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti, model_ti, num_plays = 20)
overall, fnavs, mnavs
actions_ti = analyze_actions_taken(env_ti, model_ti, plot_actions = True)

# ---- Cell 109 (run 67) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti, model_ti2)
actions_ti = analyze_actions_taken(env_ti, model_ti2, plot_actions = True)

# ---- Cell 110 (run 68) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti, model_ti3)
actions_ti = analyze_actions_taken(env_ti, model_ti3, plot_actions = True)

# ---- Cell 111 (run 69) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti, model_ti4)
actions_ti = analyze_actions_taken(env_ti, model_ti4, plot_actions = True)

# ---- Cell 112 (run 70) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti, model_ti5)
actions_ti = analyze_actions_taken(env_ti, model_ti5, plot_actions = True)

# ---- Cell 115 (run 71) ----
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# download data
ticker = yf.Ticker('TM')
df_raw = ticker.history(start = '2012-01-01', end = '2017-12-31')

# prepare the data as within the environment
df = df_raw.copy()
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.loc[:, 'returns'] = np.log(df.Close) - np.log(df.Close.shift(1))

# this is tomorrow's log-return
df.loc[:, 'returns_target'] = df.returns.shift(-1)
df.ta.strategy()

# these variables need to be deleted as they include data from the future
# this is called data leakage and creates superior results, however, is not applicable for real-life trading
df.drop(['DPO_20', 'ISA_9', 'ISB_26', 'ITS_9', 'ICS_26', 'IKS_26'], axis = 1, inplace = True)

# delete indicators with too many missing values
cols_to_drop = df.columns[df.isnull().mean() > 0.05]
df.drop(cols_to_drop, axis = 1, inplace = True)
# delete rows with missing data
df.dropna(inplace = True)

most_corr = df.corr(method = 'spearman').returns_target.abs().sort_values(ascending = False)[1:11].index.to_list()
# use this for the use_variables argument when defining an environment
most_corr

# ---- Cell 116 (run 72) ----
env_ti_corr = TradingEnvironment(ticker = "TM", start_date = '2012-01-01', end_date = '2017-12-31', use_variables = most_corr)

# ---- Cell 117 (run 73) ----
model_ti_corr5 = A2C('MlpPolicy', env_ti_corr, verbose = 0)
model_ti_corr5.learn(total_timesteps = 75000)

# ---- Cell 118 (run 74) ----
model_ti_corr4 = A2C('MlpPolicy', env_ti_corr, verbose = 0)
model_ti_corr4.learn(total_timesteps = 75000)

# ---- Cell 119 (run 75) ----
model_ti_corr3 = A2C('MlpPolicy', env_ti_corr, verbose = 0)
model_ti_corr3.learn(total_timesteps = 75000)

# ---- Cell 120 (run 76) ----
model_ti_corr2 = A2C('MlpPolicy', env_ti_corr, verbose = 0)
model_ti_corr2.learn(total_timesteps = 75000)

# ---- Cell 121 (run 77) ----
model_ti_corr1 = A2C('MlpPolicy', env_ti_corr, verbose = 0)
model_ti_corr1.learn(total_timesteps = 75000)

# ---- Cell 122 (run 78) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti_corr, model_ti_corr1)
actions_ti_corr = analyze_actions_taken(env_ti_corr, model_ti_corr1, plot_actions = True)

# ---- Cell 123 (run 79) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti_corr, model_ti_corr2)
actions_ti_corr = analyze_actions_taken(env_ti_corr, model_ti_corr2, plot_actions = True)

# ---- Cell 124 (run 80) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti_corr, model_ti_corr3)
actions_ti_corr = analyze_actions_taken(env_ti_corr, model_ti_corr3, plot_actions = True)

# ---- Cell 125 (run 81) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti_corr, model_ti_corr4)
actions_ti_corr = analyze_actions_taken(env_ti_corr, model_ti_corr4, plot_actions = True)

# ---- Cell 126 (run 82) ----
overall, fnavs, mnavs = ai_trade_performance(env_ti_corr, model_ti_corr5)
actions_ti_corr = analyze_actions_taken(env_ti_corr, model_ti_corr5, plot_actions = True)

# ---- Cell 129 (run 83) ----
from aif_environment import TradingEnvironment
from aif_analysis import ai_trade_performance, analyze_actions_taken
from stable_baselines3 import A2C, PPO
import numpy as np

env_train1 = TradingEnvironment(ticker = "TM",
    start_date = '2015-01-01',
    end_date = '2018-12-31'
)

env_test1 = TradingEnvironment(ticker = "TM",
    use_variables = env_train1.data_source.df.columns[1:].to_list(),
    scaler = env_train1.data_source.scaler,
    start_date = '2012-01-01',
    end_date = '2014-12-31'
)

model_train_test1 = A2C('MlpPolicy', env_train1, verbose = 0)
model_train_test1.learn(total_timesteps = 75000)

overall, fnavs, mnavs = ai_trade_performance(env_train1, model_train_test1)
overall, fnavs, mnavs = ai_trade_performance(env_test1, model_train_test1)

actions_training = analyze_actions_taken(env_train1, model_train_test1, plot_actions = True)
actions_test = analyze_actions_taken(env_test1, model_train_test1, plot_actions = True)

# ---- Cell 130 (run 84) ----
env_train2 = TradingEnvironment(ticker = "TM",
    start_date = '2016-01-01',
    end_date = '2019-12-31'
)

env_test2 = TradingEnvironment(ticker = "TM",
    use_variables = env_train2.data_source.df.columns[1:].to_list(),
    scaler = env_train2.data_source.scaler,
    start_date = '2013-01-01',
    end_date = '2015-12-31'
)

model_train_test2 = A2C('MlpPolicy', env_train2, verbose = 0)
model_train_test2.learn(total_timesteps = 75000)

overall, fnavs, mnavs = ai_trade_performance(env_train2, model_train_test2)
overall, fnavs, mnavs = ai_trade_performance(env_test2, model_train_test2)

actions_training = analyze_actions_taken(env_train2, model_train_test2, plot_actions = True)
actions_test = analyze_actions_taken(env_test2, model_train_test2, plot_actions = True)

# ---- Cell 131 (run 85) ----
env_train3 = TradingEnvironment(ticker = "TM",
    start_date = '2014-01-01',
    end_date = '2017-12-31'
)

env_test3 = TradingEnvironment(ticker = "TM",
    use_variables = env_train3.data_source.df.columns[1:].to_list(),
    scaler = env_train3.data_source.scaler,
    start_date = '2017-01-01',
    end_date = '2019-12-31'
)

model_train_test3 = A2C('MlpPolicy', env_train3, verbose = 0)
model_train_test3.learn(total_timesteps = 75000)

overall, fnavs, mnavs = ai_trade_performance(env_train3, model_train_test3)
overall, fnavs, mnavs = ai_trade_performance(env_test3, model_train_test3)

actions_training = analyze_actions_taken(env_train3, model_train_test3, plot_actions = True)
actions_test = analyze_actions_taken(env_test3, model_train_test3, plot_actions = True)

# ---- Cell 132 (run 86) ----
env_train4 = TradingEnvironment(ticker = "TM",
    start_date = '2012-01-01',
    end_date = '2015-12-31'
)

env_test4 = TradingEnvironment(ticker = "TM",
    use_variables = env_train4.data_source.df.columns[1:].to_list(),
    scaler = env_train4.data_source.scaler,
    start_date = '2016-01-01',
    end_date = '2018-12-31'
)

model_train_test4 = A2C('MlpPolicy', env_train4, verbose = 0)
model_train_test4.learn(total_timesteps = 75000)

overall, fnavs, mnavs = ai_trade_performance(env_train4, model_train_test4)
overall, fnavs, mnavs = ai_trade_performance(env_test4, model_train_test4)

actions_training = analyze_actions_taken(env_train4, model_train_test4, plot_actions = True)
actions_test = analyze_actions_taken(env_test4, model_train_test4, plot_actions = True)

# ---- Cell 133 (run 87) ----
env_train5 = TradingEnvironment(ticker = "TM",
    start_date = '2016-01-01',
    end_date = '2019-12-31'
)

env_test5 = TradingEnvironment(ticker = "TM",
    use_variables = env_train5.data_source.df.columns[1:].to_list(),
    scaler = env_train5.data_source.scaler,
    start_date = '2013-01-01',
    end_date = '2015-12-31'
)

model_train_test5 = A2C('MlpPolicy', env_train5, verbose = 0)
model_train_test5.learn(total_timesteps = 75000)

overall, fnavs, mnavs = ai_trade_performance(env_train5, model_train_test5)
overall, fnavs, mnavs = ai_trade_performance(env_test5, model_train_test5)

actions_training = analyze_actions_taken(env_train5, model_train_test5, plot_actions = True)
actions_test = analyze_actions_taken(env_test5, model_train_test5, plot_actions = True)
