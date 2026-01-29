import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from backtesting import Backtest, Strategy

data = yf.download("BTC-USD", period="5y")
data.reset_index(inplace=True)

signals = [0] * len(data)

for i in range(60, len(data)-1):
    train = data['Close'].iloc[:i]
    try:
        model = ARIMA(train, order=(5,1,0))
        forecast = model.fit().forecast()[0]
    except:
        forecast = train.iloc[-1]

    if forecast > data['Close'][i]:
        signals[i] = 1
    else:
        signals[i] = 0

data['signal'] = signals

class ARIMAStrategy(Strategy):
    def next(self):
        if self.data.signal[-1] == 1:
            self.buy()
        else:
            self.position.close()

bt = Backtest(data, ARIMAStrategy, cash=10000)
stats = bt.run()
print(stats)
bt.plot()
