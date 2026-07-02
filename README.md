# Trading Dissected: Companion Code

Companion code for the [Trading Dissected](https://luismatteo1.substack.com/s/trading-dissected) article series. The early parts of the series are conceptual, so the repository currently holds reproducible experiments; as the series moves from theory to application, reference implementations and tools will join them. Experiment scripts are self-contained, use fixed random seeds, and print the exact figures quoted in the articles.

## Part 4: Backtesting

`experiments_part4.py` contains two experiments.

**Experiment 1 (Part 4a): the touch-fill illusion.** A price path is simulated at one-minute resolution and aggregated to hourly candles. The same mean reversion rule is backtested twice on the same path: once filling limit orders whenever the candle low touches the level, once requiring price to trade through it. The gap between the two results measures how much of a backtest can rest on the fill assumption alone.

**Experiment 2 (Part 4b): the best of 100 worthless configurations.** One hundred configurations with zero true edge are applied to two years of simulated daily returns. The maximum observed Sharpe ratio shows what selection alone produces from noise, and matches the expected-maximum benchmark (SR*) used in the Deflated Sharpe Ratio.

## Running

```
pip install numpy
python experiments_part4.py
```

Expected output (fixed seeds):

```
EXPERIMENT 1: touch fill vs trade-through fill
optimistic (touch)     trades=  78  win=87.2%  avg/trade=+0.717%  total=+74.2%  Sharpe(ann)=4.10
realistic (through)    trades=  43  win=88.4%  avg/trade=+0.614%  total=+29.9%  Sharpe(ann)=2.79
phantom trades (touched, never traded through): 35
  their win rate: 85.7%, avg return: +0.845%

EXPERIMENT 2: best of 100 worthless configurations
single path:  median Sharpe -0.11, best Sharpe +2.32
across 500 independent paths, best-of-100 Sharpe:
  median 1.71, 5th-95th pct [1.38, 2.29]
  share of paths where best-of-100 > 1.0: 100.0%
  share of paths where best-of-100 > 1.4: 93.4%

EXPERIMENT 1 ROBUSTNESS: 25 seeds, margins 0.05%, 0.10%, 0.20%
margin 0.05%:  realistic/optimistic total return median 0.67, range [0.46, 0.79]  |  phantom share median 25.0%
  optimistic overstates the result in 25/25 seeds; phantom trades beat the realistic average in 16/25 seeds
margin 0.10%:  realistic/optimistic total return median 0.48, range [0.28, 0.66]  |  phantom share median 43.2%
  optimistic overstates the result in 25/25 seeds; phantom trades beat the realistic average in 19/25 seeds
margin 0.20%:  realistic/optimistic total return median 0.23, range [0.02, 0.41]  |  phantom share median 65.4%
  optimistic overstates the result in 25/25 seeds; phantom trades beat the realistic average in 21/25 seeds
```

The robustness block repeats Experiment 1 across 25 independent seeds and three trade-through margins. The touch convention overstates the result in all 75 seed-margin combinations.

Note: NumPy guarantees stable random streams within a major version. The figures above were produced with NumPy 1.26/2.x generators; if a future NumPy release changes the bit generator streams, absolute numbers may shift slightly while every qualitative conclusion remains.

## Structure

One experiment script per article part, named `experiments_partN.py`. Each is self-contained and runnable on its own. New scripts are added as the corresponding articles are published. Reference implementations and tools from the application-oriented parts of the series will live in their own clearly named files alongside them.

## License

MIT, see [LICENSE](LICENSE). Use, modify, and rerun freely. If the experiments change your mind about one of your own backtests, that is the intended effect.
