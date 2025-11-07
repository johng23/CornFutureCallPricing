# U.S. Corn Future Call Option Price Prediction

## What are the future and call options we consider?
This project focuses on one of the most widely traded agricultural derivatives in the world: Options on U.S. Corn Futures, which are traded on the Chicago Board of Trade (CBOT).

* A Corn Futures Contract is a standardized agreement to buy or sell 5,000 bushels of U.S. No. 2 Yellow Corn at a predetermined price on a future date. It is the primary tool used by farmers, producers, and traders to manage price risk and speculate on the future value of corn.
* An American Call Option on this Future grants the holder the right, but not the obligation, to buy one of these 5,000-bushel corn futures contracts at a specified price (the "strike price") anytime before the option's expiration date.


We analyze the prices of these call options, specifically for the March 2026 (ZCH26) contract, to understand how factors like market dynamics and weather data influence their value.

## Project Goal
This is a personal project for the Erdos Institute 2025 Spring Quantitative Finance Bootcamp.

Due to data limitations, we use the daily closing price of the nearest-to-expiry futures contracts as well as weather data to obtain volatility models, which we then use to predict prices of call options on corn futures.

## Data Sources

* 📈 YFinance API – Daily market data for futures contracts (price, volume).
* 🌾 USDA/NASS QuickStats API – Agricultural production and crop condition reports.
* ☁️ NOAA Weather API, ACIS API – Historical and real-time weather and climate data.
* 💵 FRED (Federal Reserve Economic Data) – Macroeconomic indicators such as interest rates and inflation metrics.

## Acknowledgments

Some of the code in this project is derived from a previous bootcamp project, linked [here](https://github.com/TianhaoW/ErdosAgriDerivPredict).