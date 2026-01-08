#!/usr/bin/env python3
"""
Object-Oriented BenPlan - Financial Analysis Tool

This module provides an object-oriented implementation of the BenPlan financial analysis system.
It processes transaction data from multiple sources, categorizes expenses, and generates reports.

Classes:
    - BenPlanProcessor: Main orchestrator class
    - DataLoader: Handles loading and processing transaction data
    - DataCleaner: Handles data cleaning and categorization
    - ReportGenerator: Handles report generation and Excel output
    - VisualizationManager: Handles plotting and visualization
    - ConfigurationManager: Handles configuration and settings
"""

import asyncio
import datetime as dt
import os
import re
import sys
import traceback
from multiprocessing import Pipe, Process
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from logger import logger
from compare_w_envision import compare_w_envision
                              
from utils import (get_prog_name, make_dollar_pretty, 
                   month_2_year, month_2_age, get_month_list, top_n_summary, quarterly_expenses,
                   mk_trailing_df)
from plot_functions import lineplot_df, plot_with_rails
from async_write import write_nice_df_2_xl_pipe, double_write
from configuration_manager_class import ConfigurationManager
from data_loader_class import DataLoader
from data_cleaner_class import DataCleaner

   
class BenPlanProcessor:
    """Main orchestrator class for BenPlan processing.
    Each processing step is a separate function returning a DataFrame
    the main function then calls the plotting functions - if applicable
    and calls the async writing functions
    """

    def __init__(self, cmd_line: List[str]):
        self.prog_name = get_prog_name()
        self.config_manager = ConfigurationManager(self.prog_name, cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.data_loader = DataLoader(self.config_manager)
        self.data_cleaner = DataCleaner(self.config_manager)
        conn1 = None
        self.conn2 = None
        self.write_process = None   
        self.open_files = set()
        self.task_list = []
        self.month_list = []  # place holder
        self.bp_simple_keep = self.config['BP_SIMPLE_KEEP']
        self.discretionary_cat = self.config['DISCRETIONARY_CAT']

    def _log_info_tick(self, msg: str, xtra_log_msg = None) -> None:
        """Log info and tick timer"""
        if xtra_log_msg is not None:
            logger.info(f"{msg}\n{xtra_log_msg}")
        else:
            logger.info(msg)
        self.config_manager._tick_timer(msg)
        return

    
    async def process(self) -> None:
        """Main processing method."""

        # TODO: move this to the ConfigurationManager class
        # Get current directory and setup paths
        paths = self.config_manager.output_paths
        # Setup output files
        self._open_output_files()
        # Setup async writing
        # conn1 is the pipe to the write process (send data to it)
        # write_process is the process that receives the data via conn2 # and writes it to the Excel file
        conn1, conn2 = Pipe()
        self.write_process = Process(target=write_nice_df_2_xl_pipe, args=(conn2,))
        self.write_process.start()
        xl_outf = self.config_manager.output_paths['xl_out_filename']
        self.xl_outf = xl_outf  # FIXME: this is a hack to get the xl_outf into the class


        # Load and process data
        cumul_df = self.data_loader.load_all_transaction_data()
        self._log_info_tick(f"Loaded {len(cumul_df.index):,} transactions")

        # Clean and prepare data
        clean_df = self.data_cleaner.clean_and_prepare_data(cumul_df)
        self._log_info_tick(f"Cleaned {len(clean_df.index):,} transactions")

        # Write combined data to Excel
        task = double_write(conn1, xl_outf,  clean_df, "Combined", False, None, quick_flag=True)
        self.task_list.append(task)
        self._log_info_tick(f"Wrote {len(clean_df.index):,} transactions to Combined sheet")
        

        # Handle BenPlan Categories
        agg, pivot = self._get_benplan_categories(conn1, clean_df)
        benplan_cat_df = pivot.copy(deep=True)  # preserve the original pivot
        self.month_list = get_month_list(pivot)   # list of all the months
        nb_month = len(self.month_list)
        task = double_write(conn1, xl_outf,  agg, 'BenPlan Categories Aggr', False, None, quick_flag=True)
        self.task_list.append(task)
        task = double_write(conn1, xl_outf,  pivot, 'BenPlan Categories 1', True, 2, quick_flag=True)
        self.task_list.append(task)
        self._log_info_tick(f"Done processing: BenPlan averages", f"{make_dollar_pretty(pd.DataFrame(pivot[['YearlyAvgAll', 'Last12Mo']]))}")
        self._log_info_tick(f"Done processing BenPlan categories - {nb_month} months")

        # Create chart for Ignore and Transfer categories - to confirm that they are close to 0
        ignore_transfer_df = self._process_ignore_transfer(pivot)  # just creates a chart, no need to save data 

        # Create charts for YoY and trailing 12 months
        lag_df, yoy_lag_df = self._process_trailing_12_months(pivot)
        task = double_write(conn1, xl_outf,  lag_df, 'BenPlan - Trailing 12 months', True, None, quick_flag=True)
        self.task_list.append(task)
        task = double_write(conn1, xl_outf,  yoy_lag_df, 'BenPlan YoY Trailing 12 months', True, None, quick_flag=True)
        self.task_list.append(task)
        self._log_info_tick(f"BenPlan Expenses YoY Trailing 12 months", f"{make_dollar_pretty(lag_df)}")
        bp_simpl_df = self._process_simplified_categories(pivot)
        task = double_write(conn1, xl_outf,  bp_simpl_df, 'BenPlan Simplified', True, None, quick_flag=True)
        self.task_list.append(task)
        self._log_info_tick("BenPlan Simplified", f"{make_dollar_pretty(bp_simpl_df)}")
        
        # --- Compute BenPlan expenses by Year (1) calendar (2) Age
        ben_df = self._process_benplan(pivot)
        tsk = double_write(conn1, xl_outf,  ben_df, "BenPlan", True, None, quick_flag=True)
        self.task_list.append(tsk)
        self._log_info_tick(f"BenPlan Expenses YoY Trailing 12 months",f"{make_dollar_pretty(lag_df)}")

        # Plot XTRA
        xtra_df = self._plot_xtra(pivot)

        # ----- Compare with Envision Projections
        # Use the latest worksheet in the Envision file
        delta_df = compare_w_envision(pivot, self.age_lst, self.config_manager.output_paths['out_file'])
        tsk = double_write(conn1, xl_outf,  delta_df, "Envision", True, None, quick_flag=True)
        self.task_list.append(tsk)
        self._log_info_tick(f"Envision",f"{make_dollar_pretty(delta_df)}")

        # Create a new summary of monthly expenses
        monthly_expenses_df = self._process_monthly_expenses(clean_df)
        tsk = double_write(conn1, xl_outf,  monthly_expenses_df, "Monthly Expenses", True, None, quick_flag=True)
        self.task_list.append(tsk)

        # Create a new summary array with top N category
        top_n_summary(monthly_expenses_df, "Ongoing Expenses", self.config['WINDOW'], self.plt_file)
        qtrly_df = quarterly_expenses(monthly_expenses_df, self.plt_file)
        self._log_info_tick(f"Quarterly Expenses", f"{make_dollar_pretty(qtrly_df)}")

        # Break down expenses in Rtmt_Spending
        rtmt_spending_df = self._process_rtmt_spending(clean_df)
        tsk = double_write(conn1, xl_outf,  rtmt_spending_df, "Rtmt_Spending Expenses", True, None, quick_flag=True)
        self.task_list.append(tsk)
        self._log_info_tick(f"Rtmt_Spending Expenses", f"{make_dollar_pretty(rtmt_spending_df)}")

        # Compute TopN Rtmt_Spending Expenses
        top_n_summary(rtmt_spending_df, "Rtmt_Spending Expenses", self.config['WINDOW'], self.plt_file)

        # Compute Travel expenses
        travel_df = self._process_travel(clean_df)
        tsk = double_write(conn1, xl_outf,  travel_df, "Travel", True, 3, quick_flag=True)
        self.task_list.append(tsk)
        self._log_info_tick(f"Travel", f"{make_dollar_pretty(travel_df)}")
       
        # Compute Discretionary expenses
        agg_disc, pivot_discr = self._process_discretionary(conn1, clean_df)
        task = double_write(conn1, xl_outf,  agg_disc, 'Discretionary Expenses Aggr', False, None, quick_flag=True)
        self.task_list.append(task)
        task = double_write(conn1, xl_outf,  pivot_discr, 'Discretionary Expenses', True, 2, quick_flag=True)
        self.task_list.append(task)
        self._log_info_tick(f"Discretionary", f"{make_dollar_pretty(pivot_discr[['YearlyAvgAll', 'Last12Mo']])}")


        fyi_data = self.compute_fyi_data(benplan_cat_df)
        self._log_info_tick(f"FYI Data", f"{make_dollar_pretty(fyi_data)}")
        task = double_write(conn1, xl_outf,  fyi_data, 'FYI Data', True, None, quick_flag=True)
        self.task_list.append(task)

        
        # Wrap up
        try:
            self.plt_file.close()  # don't wait for async stuff 
            logger.info("--> PDF Plot File Ready")
        except Exception as e:
            logger.error(f"Error closing plot file: {e}")   

        # None filename -> end, other arguments are ignored
        tsk = double_write(conn1, None,  pd.DataFrame(), "", False, None, quick_flag=True)
        self.task_list.append(tsk)
        logger.info(f"Gathering task_list: {len(self.task_list)} tasks")
        
        await asyncio.gather(*self.task_list)  # Wait for the async tasks to complete
        self.write_process.join()  # Wait for the write process to finish
        
        # Clean up resources
        await self.cleanup(kill_flag=False)
        
        logger.info(f"{self.prog_name}: Processing DONE")
        return
    

    def _open_output_files(self) -> None:
        """Setup output files and directories."""
        paths = self.config_manager.output_paths
        # Handle Excel output file
        if os.path.exists(paths['xl_out_filename']):
            renamed_file = re.sub('\\.xlsx', '_Saved.xlsx', paths['xl_out_filename'])
            os.replace(paths['xl_out_filename'], renamed_file)
        self.xl_out_filename = paths['xl_out_filename']
        # Don't add string paths to open_files - only add actual file objects

        # Create quick directory
        if not os.path.exists(paths['quick_dir']):
            os.makedirs(paths['quick_dir'])

        # Setup output file
        self.outf = open(paths['out_file'], mode='w', encoding='utf-8')
        self.open_files.add(self.outf)

        # Setup plot file
        self.plt_file = PdfPages(paths['plt_file'])
        self.open_files.add(self.plt_file)

        logger.info(
                'Processing files from: ' + os.path.basename(os.path.dirname(self.xl_out_filename)))
        return

    def _get_benplan_categories(self, conn, in_df: pd.DataFrame) -> pd.DataFrame:
        agg, pivot = self._agg_by_category(conn, in_df, 'BenPlan')
        return agg, pivot
    
    def _process_ignore_transfer(self, pivot: pd.DataFrame) -> None:
        """ Compute Ignore and Transfer categories and make sure they are close to 0 """
        new_pivot_data = pivot.loc[['Ignore', 'Transfer'], :]
        lineplot_df(new_pivot_data.iloc[:, 0:-2], title='BenPlan Ignore & Transfer Categories',
                    plt_file=self.plt_file, show_mean='Mean')
        return new_pivot_data.iloc[:, 0:-2]

    def _process_simplified_categories(self, pivot: pd.DataFrame) -> None:
        """ Plot simplified categories """
        bp_simple_keep = self.bp_simple_keep
        # Checking:
        missing = [x for x in bp_simple_keep if x not in pivot.index]
        if len(missing) > 0:
            logger.warning(f"WARNING: the following BenPlan Main Categories are missing: {missing}")
        bp_simpl_other = [x for x in list(pivot.index) if x not in bp_simple_keep]
        # Note that pivot_data is Not sorted -> we cannot use int indices
        new_pivot_data = pivot.loc[bp_simple_keep, :]
        # tpivot_other_data DF contains low variance categories
        pivot_other_data = pivot.loc[bp_simpl_other, :]
        other_sum = pivot_other_data.sum(axis=0)
        new_pivot_data.loc['Other'] = other_sum
        self._log_info_tick(f"Simplified Monthly Expenses by BenPlan Categories", f"{make_dollar_pretty(new_pivot_data)}")
        lineplot_df(new_pivot_data[self.month_list],
                    title='Simplified Monthly Expenses by BenPlan Categories', plt_file=self.plt_file)

        new_pivot_data = pivot.loc[['XTRA', 'Rtmt_Spending', 'Travel'], :]
        lineplot_df(new_pivot_data.iloc[:, 0:-2], title='XTRA, Rtmt_Spending & Travel Categories',
                    plt_file=self.plt_file, show_mean='Last12Mo')
        return new_pivot_data


    def _process_trailing_12_months(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """ Make plots of trailing 12-month expenses by categories - exclude last 2
        columns YearlyAvgAll, Last12Mo """
        lag_df = mk_trailing_df(in_df.iloc[:, :-2], lag=12)

        # Simplify: just show YoY trailing 12 months
        # Get the month labels of every year that match this month: e.g. 2015-07, 2016-07, ... ,
        # 2020-07
        lag_df_col = list(lag_df.columns)
        # Remove aggregate columns: YearlyAvgAll, Last12Mo - if they are present
        lag_df_col = [x for x in lag_df_col if x not in ['YearlyAvgAll', 'Last12Mo']]
        this_month = lag_df_col[len(lag_df_col) - 1].split('-')[1]  # e.g. '07'

        yoy_lag_df_col = [x for x in lag_df_col if
                          x.split('-')[1] == this_month]  # e.g. ['2015-07', ... , '2020-07']
        yoy_lag_df = pd.DataFrame(lag_df[yoy_lag_df_col])  # sub-sampled DF
        self._log_info_tick(f"BenPlan Expenses YoY Trailing 12 months", f"{make_dollar_pretty(yoy_lag_df)}")


        lineplot_df(yoy_lag_df.loc[self.bp_simple_keep],
                    title='Simplified BenPlan Categories - YoY 12 months',
                    plt_file=self.plt_file)

        return lag_df, yoy_lag_df


    def _process_benplan(self, pivot: pd.DataFrame) -> None:
        """ Create BenPlan Expenses by Year and Age """
        month_list = self.month_list

        # --- Compute BenPlan expenses by Year
        # Exclude the aggregate columns
        year_set = set(map(month_2_year, month_list))
        year_set = sorted(year_set)
        # for each year - sum the values for the months in that year
        lst_year = max(year_set)
        for yr in year_set:
            # Make a list of the months in that year
            mm_list = [x for x in month_list if month_2_year(x) == yr]
            # Scale proportionedly to number of months in that year
            pivot[yr] = pivot[mm_list].sum(axis=1) * 12 / len(mm_list)
            # Add a "to date" column for the last year
            if yr == lst_year:
                col_name = str(yr) + '_TD'
                pivot[col_name] = pivot[mm_list].sum(axis=1)
        tmp_df = make_dollar_pretty(pd.DataFrame(pivot[year_set]))
        self._log_info_tick('BenPlan Expenses by Year', f"{tmp_df}")

        # --- Compute BenPlan expenses by Age
        age_set = set(map(month_2_age, month_list))
        age_lst = sorted(age_set)
        self.age_lst = age_lst  # save for later use
        # for each age - sum the values for the months in that age
        oldest_age = max(age_lst)
        for yr in age_lst:
            # Make a list of the months in that age
            mm_list = [x for x in month_list if month_2_age(x) == yr]
            if yr == oldest_age:  # Use the past 12 months, since we likely have incomplete series
                # Augment mm_list backwards in order to have a total of 12 months
                # Add the months from last_age-1
                mm_list += [x for x in month_list if month_2_age(x) == (oldest_age-1)]
                # Sort mm_list, items are YYYY-MM so regular sort will work
                mm_list = sorted(mm_list)
                # Pick the last 12 months
                mm_list = mm_list[-12:]
                logger.info(f"Computing by Age - mm_list: {mm_list} for age: {oldest_age}")
                pivot[yr] = pivot[mm_list].sum(axis=1)
            else:
                # Scale proportionedly to number of months in that age
                # only the youngest age will not have 12 months
                pivot[yr] = pivot[mm_list].sum(axis=1) * 12 / len(mm_list)
            # Add a "to date" column for the last age
            if yr == oldest_age:
                col_name = str(yr) + '_TD'
                pivot[col_name] = pivot[mm_list].sum(axis=1)
        tmp_df = make_dollar_pretty(pd.DataFrame(pivot[age_lst]))
        self._log_info_tick('BenPlan Expenses by Age', f"{tmp_df}")
        del tmp_df
        # Write the columns in Excel file in reverse order for convenience
        ll = list(reversed(pivot.columns))
        ben_df = pd.DataFrame(pivot[ll]).copy(deep=True)  # deep copy for safety
        self.config_manager._tick_timer("BenPlan")
        return ben_df

    def _plot_xtra(self, pivot: pd.DataFrame) -> None:
        """ Plot XTRA expenses """
        xtra_df = pivot.loc['XTRA', self.month_list].copy(deep=True)
        plot_with_rails(xtra_df, title='Monthly XTRA Expenses', plt_file=self.plt_file)
        return xtra_df

    def _process_monthly_expenses(self, cumul_df: pd.DataFrame) -> pd.DataFrame:
        """ Process monthly expenses """
        # Create Pivot Table for 'OngoingExpenseCat' master category
        # Need to make a deep copy
        tmp_df = cumul_df[cumul_df['MasterCat'] == 'OngoingExpenseCat'].copy(deep=True)
        agg2 = pd.DataFrame(tmp_df.groupby(['Month', 'Category'], as_index=False)['Amount'].sum())
        # Turn expenses to a positive humber
        agg2['Amount'] *= -1.0

        pivot = pd.pivot_table(pd.DataFrame(agg2), index=['Category'], values=['Amount'], columns=['Month'])
        pivot.fillna(0.0, inplace=True)
        # Get rid of the multi-level index: pivot.columns.get_level_values(0) =
        # ['Amount', 'Amount', ...]
        pivot.columns = pivot.columns.get_level_values(1)
        # Compute the overall average Yearly expense
        pivot['YearlyAvgAll'] = 12.0 * pivot.mean(axis=1)
        # last 12 months average
        pivot['Last12Mo'] = pivot.iloc[:, -13:-1].sum(axis=1)  # Skip YearlyAvgAll column
        pivot.sort_values('Last12Mo', axis=0, ascending=False, inplace=True, na_position='last')
        self.config_manager._tick_timer("Monthly Expenses")
        return pivot

    def _process_rtmt_spending(self, clean_df: pd.DataFrame) -> None:
        """ Break down expenses in Rtmt_Spending """
        tmp_df = clean_df[clean_df['BenPlan'] == 'Rtmt_Spending'].copy(deep=True)
        agg2 = pd.DataFrame(tmp_df.groupby(['Month', 'Category'], as_index=False)['Amount'].sum())
        # Turn expenses to a positive humber
        agg2['Amount'] *= -1.0

        pivot = pd.pivot_table(agg2, index=['Category'], values=['Amount'], columns=['Month'])
        # Fill in NA with $0
        pivot.fillna(0.0, inplace=True)
        # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount', 'Amount',
        # ...]
        pivot.columns = pivot.columns.get_level_values(1)
        # Compute the overall average Yearly expense
        pivot['YearlyAvgAll'] = 12.0 * pivot.mean(axis=1)
        # Trailing 12 months
        pivot['Last12Mo'] = pivot.iloc[:, -13:-1].sum(axis=1)  # Skip YearlyAvgAll column
        pivot.sort_values('Last12Mo', axis=0, ascending=False, inplace=True, na_position='last')
        return pivot

    def _process_travel(self, clean_df: pd.DataFrame) -> None:
        travel_df = clean_df[clean_df['BenPlan'] == 'Travel'].copy(deep=True)
        travel_df.drop(columns=['Payee', 'Category', 'Source', 'MasterCat', 'BenPlan', 'Discretionary',
                                'Memo/Notes'], inplace=True)
        # Rename the "Tags" column to Trip
        travel_df.columns = list(map(lambda x: x if x != "Tags" else "Trip", list(travel_df.columns)))
        # Replace blank tags with Other
        travel_df['Trip'] = pd.Series(travel_df['Trip']).apply(lambda x: 'Other' if x == '' else x)
        trip_set = set(travel_df['Trip'])
        logger.info(f"Number of Trips:  {len(trip_set)}")
        travel_agg = pd.DataFrame(travel_df.groupby(['Month', 'Trip'], as_index=False)['Amount'].sum())
        travel_pivot = pd.pivot_table(pd.DataFrame(travel_agg), index=['Trip'], values=['Amount'], columns=['Month'])
        # Fill in NA with $0
        travel_pivot.fillna(0.0, inplace=True)
        # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount', 'Amount',
        # ...]
        travel_pivot.columns = travel_pivot.columns.get_level_values(1)
        travel_pivot['Total'] = travel_pivot.apply(sum, axis=1)

        # ToDo - Move this earlier - before any computations are performed - otherwise it breaks Cost
        #  per Day computation
        travel_cat_df = self.data_loader.get_travel_cat_df()
        
        vacation_set = set(travel_cat_df['Vacation'])
        if vacation_set != trip_set:
            if len(trip_set - vacation_set) > 0:
                logger.info(f"Unexpected Trips: {trip_set - vacation_set}")
            if len(vacation_set - trip_set) > 0:
                logger.info(f"Missing Trips: {vacation_set - trip_set}")

        # Add #days for each trip and average cost per day
        day_map = dict(zip(travel_cat_df['Vacation'], travel_cat_df['#Days']))
        travel_pivot['Days'] = [int(day_map[x]) for x in travel_pivot.index]
        travel_pivot['CostpDay'] = - travel_pivot['Total'] / travel_pivot['Days']
        travel_pivot.sort_values(by='CostpDay', ascending=False, inplace=True)
        
        return travel_pivot


    def _process_discretionary(self, conn: Pipe, clean_df: pd.DataFrame) -> None:
        """ Process discretionary expenses """
        agg, pivot = self._agg_by_category(conn, clean_df, 'Discretionary')
        # plot only the selected categories - which are in rows
        plot_df = pivot.loc[pivot.index.isin(self.discretionary_cat)].copy(deep=True)
        lineplot_df(plot_df.iloc[:, 0:-2], title='Discretionary & Mandatory Spending',
                plt_file=self.plt_file, show_mean='Last12Mo')

        return agg, pivot


    def _agg_by_category(self, conn: Pipe, df: pd.DataFrame, master_cat: str) -> pd.DataFrame:
        """ 
        Aggregate the data by category and month
        :param df: input DF
        :param master_cat: Category column to aggregate by
        :param title: Title of the output tab
        :param xl_out_filename: Excel file to write to
        :param outfile: output file to print to
        :param conn: connection to the async process
        :return: DF of amounts in categories (row) by month (col)
        """
        agg = pd.DataFrame(df.groupby(['Month', master_cat], as_index=False)['Amount'].sum())

        # Group by category and month, sum amounts
        # pivot = df.pivot_table(values='Amount', index=[master_cat], columns='Month', aggfunc='sum',
        #                        fill_value=0)
        pivot = pd.pivot_table(agg, index=[master_cat], values=['Amount'], columns=['Month'], fill_value=0.0)

        # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount',
        # 'Amount', ...]
        pivot.columns = pivot.columns.get_level_values(1)

        # Compute the overall average Yearly expense
        pivot['YearlyAvgAll'] = 12.0 * pivot.mean(axis=1)
        # last 12 months average - i.e. sum of last 12
        pivot['Last12Mo'] = pivot.iloc[:, -13:-1].sum(axis=1)  # Skip YearlyAvgAll column


        return agg, pivot

    def _generate_visualizations(self, pivot: pd.DataFrame, conn: Pipe) -> None:
        """Generate visualizations."""
        # Create trailing 12-month analysis
        month_list = [x for x in pivot.columns if x not in ['YearlyAvgAll', 'Last12Mo']]
        nb_month = len(month_list)
        logger.info(f'\n#Months: {nb_month}')

        # Plot ignore and transfer categories
        ignore_transfer_data = pivot.loc[['Ignore', 'Transfer'], :]
        lineplot_df(ignore_transfer_data.iloc[:, 0:-2],
                                               title='BenPlan Ignore & Transfer Categories',
                                               plt_file=self.plt_file, show_mean='Mean')

    def compute_fyi_data(self, pivot: pd.DataFrame) -> None:
        """FYI data:
        Aggregate VC Contributions: VC_Invest
        Aggregate VC Distributions: VC_Principal
        Aggregate Vanguard Contributions: Vanguard
        Aggregate Kayla Spend: Kayla
        """
        
        # FYI: need to slide in 2 steps because the columns are not numeric
        fyi_data = pivot.loc[self.config['FYI_DATA']].iloc[:, :-2].copy(deep=True) # Skip the last 2 columns
        # FYI: fyi_data.sum(skipna=True) is not working
        fyi_data['Total'] = fyi_data.apply(lambda x: x.sum(skipna=True), axis=1)
        # Write fyi_data to a  Excel file
        fyi_data.to_excel('fyi_data.xlsx', index=True)  
        return fyi_data['Total']


    async def cleanup(self, kill_flag: bool = False) -> None:
        """
        Cleanup resources.
        1. Send stop signal to write process
        2. If kill_flag is True (KeyboardInterrupt or Exception), kill the write process i.e. async tasks
        3. Close open files
        4. Stop timer
        """
        
        try:
            # Signal write process to stop
            logger.info(f"Cleanup: sending stop signal to write process")
            if hasattr(self, 'conn1') and self.conn1:
                self.conn1.send(None)
        except (BrokenPipeError, ConnectionResetError):
            # Pipe is already closed, which is fine
            logger.debug("Pipe already closed during cleanup")
        except Exception as e:
            logger.error(f"Error sending stop signal to write process: {e}")

        write_process = self.write_process

        if kill_flag:
            logger.info("Killing write process")
            if write_process and write_process.is_alive():
                write_process.terminate()
                write_process.join(timeout=2)
                if write_process.is_alive():
                    logger.warning("Write process did not terminate gracefully, forcing termination")
                    write_process.terminate()
                    write_process.join(timeout=2)   
        else:
            # Wait for write process to finish
            logger.info(f"Cleanup: waiting for write process to finish")
            if write_process and write_process.is_alive():
                write_process.join(timeout=300)  # Wait up to 5  minutes
                if write_process.is_alive():
                    logger.error("Write process did not finish in 300 seconds, forcing termination")
                    write_process.terminate()
                    write_process.join(timeout=2)
                    if write_process.is_alive():
                        write_process.kill()
                        write_process.join()

        # Close open files
        logger.info(f"Cleanup: closing open files: {len(self.open_files)}")
        for file in self.open_files:
            try:
                file.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")

        # Final check to ensure write process is terminated
        if hasattr(self, 'write_process') and self.write_process and self.write_process.is_alive():
            logger.warning("Write process still alive after cleanup, forcing termination")
            try:
                self.write_process.kill()
                self.write_process.join(timeout=1)
            except Exception as e:
                logger.error(f"Error killing write process: {e}")

        # Close the pipe connection after write process is done
        if hasattr(self, 'conn1') and self.conn1:
            try:
                self.conn1.close()
            except Exception as e:
                logger.warning(f"Error closing pipe connection: {e}")

        if not kill_flag:
            self.config_manager._stop_timer()  # prints the timer intervals
        logger.info(f"Cleanup: cleanup completed")
        return


async def main(cmd_line: List[str]):
    """Main entry point for BenPlan.
    processor_kill_flag: if True, async processes will be killed by the cleanup function
    """
    processor = BenPlanProcessor(cmd_line)
    processor_kill_flag = False
    exit_code = 0
    # Start the timer for the entire program
    start_time = dt.datetime.now()
    try:
        await processor.process()
        logger.info(f"Processing completed")
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Cleaning up...")
        # Cancel all async tasks
        processor_kill_flag = True
        exit_code = 1
    except Exception as e:
        logger.info("\nError: {e}")
        # dump stack trace
        logger.error(traceback.format_exc())
        logger.error("\n----\n")
        processor_kill_flag = True
        exit_code = -2
    finally:
        # Only call cleanup if there was an exception or interrupt
        if processor_kill_flag:
            try:
                logger.info(f"Starting final cleanup - kill_flag: {processor_kill_flag}")
                await processor.cleanup(processor_kill_flag)
            except Exception as cleanup_error:
                logger.error(f"Error during final cleanup: {cleanup_error}")
    end_time = dt.datetime.now()
    logger.info(f'End: {str(end_time)} -- Run Time: {str(end_time - start_time)}\n')
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
