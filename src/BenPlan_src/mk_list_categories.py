#!/usr/bin/env python

""" Read all account files and create comprehensive list of categories """

import csv
import os
import re
import time
import numpy as np
import pandas as pd
import expenses_utilities as ut

# Configure the settings
dirDateFormat = ut.Settings['dirDateFormat']
qDateFormat = ut.Settings['qDateFormat']
firstDate = ut.Settings['firstDate']
masterCatFile = ut.Settings['masterCatFile']
csvDir = ut.Settings['csvDir']
dirNamePattern = ut.Settings['dirNamePattern']
col2Keep = ut.Settings['col2Keep']


# To FIX -----------------------
# Eliminate transactions before 5/15/2015
# Add total $Amount and #transactions for each cat
# Convert to a pre-processing step to reassign come categories to better labels
# Group expsenses/income for Loft & PPP
# ---------------

# ---- Main ----

# Get the list of all directories in csvDir
dFiles = [d.name for d in os.scandir(csvDir) if d.is_dir() == True]
# fullmatch returns None if there is No match
dirFiles = [d for d in dFiles if re.fullmatch(dirNamePattern,d)]

# Get the most recent directory
currentDir = max(dirFiles)
aggFile = 'aggregate-'+currentDir+'.csv'
combFile = 'combined-'+currentDir+'.csv'
avgFile = 'average-'+currentDir+'.csv'
pivotFile = 'monthly-'+currentDir+'.csv'
allpivotFile = 'monthly-all-'+currentDir+'.csv'
outfile = '../categories-'+currentDir+'.csv' # Want file in CSV dir
print('Processing files from:', currentDir)

# Get the master list of Categories
df = pd.read_csv(masterCatFile, sep=',', header=0) # column names are on row 1
# df.columns=['Category', 'MappedCat', 'masterCat', 'blank', 'masterList']
masterCatSet=set(df['Category'])

# Change directory to the most recent
os.chdir(csvDir + currentDir)
# Get a list of the CSV files - only
csvFiles = [f.name for f in os.scandir('.') if f.name.split('.')[1] == 'csv']

# suppressList = ['aggregate.csv', 'combined.csv']
suppressList = [aggFile, combFile, avgFile, pivotFile, allpivotFile]
csvFiles = [f for  f in  csvFiles if f not in suppressList]

print("Files to be processed:")
print(csvFiles)

# masterCat is the masterlist of Categories
cumulDF = pd.DataFrame(columns=['Category', 'Amount', 'Count'])
for file in csvFiles:
	print('Processing file: {}'.format(file))
	cat = ut.processFile(file)
	cumulDF = cumulDF.append(cat)

# ------ Clean up cumulDF ------

cumulDF['Month'] = cumulDF['Date'].apply(lambda x: ut.date2month(x,qDateFormat))
z = cumulDF[cumulDF.Month == 'xxxx-xx']
if len(z.index) > 0:
	print('rows with weird dates')
	print(z)
# cumulDF['Month'] = cumulDF['Date'][1:].apply(lambda x: ut.date2month(x))
# printDF(cumulDF, "Cumul")
# printDF(cumulDF, msg="Final DF", verbose=True)


# Keep the transactions in specified date range
firstMo = ut.date2month(firstDate, qDateFormat)
# Cheeky to use comparison on strings, but the label Month was built for this purpose (also for plot)
# Would have to use a time function on Date
# lastMo is in format yyyy-mo-dd - so we need to remove '-dd' without making assumptions on the length of mo or dd
lastMo = ut.date2month(currentDir, dirDateFormat)
print('First Mo: {} - Last Mo: {}'.format(firstMo, lastMo))
cumulDF = cumulDF[cumulDF.Month >= firstMo]
# Note the < ... we want to exclude the last month, since it is incomplete
cumulDF = cumulDF[cumulDF.Month < lastMo]
# printDF(cumulDF, msg="Good Dates", verbose=False)


# Convert Amount to Float so that it can be added up - Note that none of the 2 approaches below work
# E.g Converting to float pukes on commas (e.g 1,599.60)!!
# cumulDF['Amount'] =cumulDF.Amount.apply(lambda x: x.strip(','))
# cumulDF['Amount'] =cumulDF['Amount'].astype(str).replace(',','').astype(float)
# Also makes expenses a positive humber
cumulDF['Amount'] =cumulDF['Amount'].apply(lambda x: -float(x.replace(',','')))

# Flag Uncategorized transactions
tmpdf = cumulDF[cumulDF.Category == 'Uncategorized']
if len(tmpdf.index) > 0:
	print(len(tmpdf.index), 'Uncategorized Transactions:')
	print(tmpdf)
	print()

masterCat = cumulDF.groupby(['Category'],as_index=False)['Amount'].sum()
masterCat.columns = ['Category', 'Amount']
agg2 = cumulDF.groupby(['Category'],as_index=False)['Amount'].count()
agg2.columns = ['Category', 'Count']
masterCat['Count']=agg2['Count']
masterCat.sort_values('Count', ascending=True, inplace=True, na_position='last')
masterCat.to_csv(outfile, header=True)

# Check if we have new Categories
catSet = set(masterCat['Category'])
newCat = catSet - masterCatSet   # elements of catSet not in masterCatSet - should be empty
if len(newCat) > 0:  # we have new Categories
	print('*** Heads UP: we have {:d} NEW categories'.format(len(newCat)))
	print(newCat)
