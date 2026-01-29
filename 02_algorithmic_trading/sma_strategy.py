from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd

# Load BTC data
data = pd.read_csv("btc_1d_data_2018_to_2025.csv")
data['Date'] = pd.to_datetime(data['Date'])

class SMACrossoverStrategy(Strategy):
    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, 30)
        self.sma_slow = self.I(SMA, close, 60)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
            self.sell()

bt = Backtest(
    data,
    SMACrossoverStrategy,
    cash=100000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
bt.plot()
