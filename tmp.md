# Holdings Cross-Correlation

## Objective: 

Load stock data and same them in a cache file. 

## Input file

 has the format: Positions_YYYY_MM_DD.xlsx - where YYYY_MM_DD represent the creation date of the file. We need the most recent file in the ./Data folder

The asset tickers are in the first column "Ticker"

## Output

For each ticker, the daily Adjusted Close ('Adj Close') prices will be dowloaded from Yahoo Finance for the past 10 years-

Compute the daily log return and save date, adjusted close and daily log return in a JSON file in the Cache folder. Each JSON file will have the format "ticker_Creation_Date".  The cache is only refreshed if the program is run 1 month or more later than the month of "Creation_Date" - print whether the program uses cached data for all tickers. If one or more tickers need to be refreshed print the number of tickers being refreshed followed by the list.

For each ticker, print - using `logger.info` - the first available date for the data and the total number of data points in the series.

## Config File

The configuration file is a TOML file  whose name has the same root as the script. The `.toml` file contains: 

* Prefix string for the Positions file: "Positions_"
* Location of input folder: "./Data"
* Date format for both input and output file: "%Y_%m\_%d"
* Location of cache folder: "./Data/Cross_Corr_Cache"




