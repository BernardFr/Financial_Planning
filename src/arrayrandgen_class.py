#!/usr/local/bin/python3
"""
Scaffolding for creating a class 
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

# For Testing
ROR_STATS = [ (3.00, 9.00),(2.60, 7.70), (1.50, 0.90),  (-0.05, 0.0)]  # Stocks, Bonds, TBills, Cash
ASSET_CLASSES = {'Stocks': 800000,
                'Bonds': 400000,
                'TBills': 200000,
                'Cash': 100000}
MASTER_SEED = 42
NB_CPU = 4
RUN_CNT = 1000

class ArrayRandGen:
    """
    This class is used to generate a list of random numbers based on a mean and stddev
    The numbers are: 1 + random_number/100  (interest rates)
    NOTE that the RoR is clipped to the floor and ceiling to account for historical data

    Parameters:
    config_manager: ConfigurationManager object
    name: str - the name of the asset class
    mean: float - the mean of the random numbers
    stdv: float - the stddev of the random numbers
    seed: int - the seed for the random number generator
    """
    def __init__(self,config_manager: ConfigurationManager, name: str, mean: float, stdv: float, seed) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.nb_rv = self.config['end_age'] - self.config['start_age'] + 1  # number of random values to generate
        self.mean = mean
        self.stdv = stdv
        self.ror_floor = self.config['RoR_floor']
        self.ror_ceiling = self.config['Ror_ceiling']
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.name = name

    def __iter__(self):
        return self

    def __next__(self) -> list[float]:
        """
        returns a list of nb random numbers based on mean and stddev
        numbers are: 1 + random_number/100  (interest rates)
        NOTE that the RoR is clipped to the floor and ceiling to account for historical data
        """
        random_values = self.rng.normal(self.mean, self.stdv, size=self.nb_rv)
        ror_values = 1 + 0.01 * random_values
        # Clip the final RoR values to historical bounds
        ror_values = np.clip(ror_values, self.ror_floor, self.ror_ceiling)
        return ror_values.tolist()


def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    start_age = config_manager.config['start_age']
    end_age = config_manager.config['end_age']  # number of ages in the range
    nb_ages = end_age - start_age + 1

    ror_stats = ROR_STATS
    nb_assets = len(ASSET_CLASSES.keys())
    print(f"ROR Stats:\n{ror_stats}")
    stats_df = pd.DataFrame(ROR_STATS, index=ASSET_CLASSES.keys(), columns=['Mean', 'Std'])
    print(f"Stats DF:\n{stats_df}")
    for name, ror_tuple in zip(ASSET_CLASSES.keys(), ROR_STATS):
        stats_df.loc[name] = [ror_tuple[0], ror_tuple[1]]
    logger.info(f"Using Morningstar stats as is")
    # Create a sequence of pseudo-random seeds for the random number generators for each CPU
    master_seed_sequence = np.random.SeedSequence(MASTER_SEED)
    seed_sequence = master_seed_sequence.spawn(NB_CPU)
    ror_gen_list = []
    for ror_tuple, name, seed in zip(ROR_STATS, ASSET_CLASSES.keys(), seed_sequence):
        array_rand_gen = ArrayRandGen(config_manager, name, ror_tuple[0], ror_tuple[1], seed)
        ror_gen_list.append(array_rand_gen)
    # ror_lst_lst = []  # list of lists of RoR values for each asset class. Each has nb_ages * run_cnt values
    # ror_lst_lst = list([]*nb_assets) # create a list of nb_assets lists
    ror_lst_lst = [[] for _ in range(nb_assets)]
    # print(f"ROR LST LST:\n{len(ror_lst_lst)}")
    ror_df = pd.DataFrame(columns=range(RUN_CNT*nb_ages))  # need to specify the number of columns
    for nb, gener in enumerate(ror_gen_list):
        name = gener.name  # get the asset class name from the generator
        for _ in range(RUN_CNT):
            # generator returns a list which contains the list of values for each age
            new_ror_values = list(next(gener))
            ror_lst_lst[nb].extend(new_ror_values) 
        ror_df.loc[name] = pd.Series(ror_lst_lst[nb])
    # print(f"Length of ror_lst_lst[0][0]: {len(ror_lst_lst[0][0])}")
    # # merge the lists in ror_lst_lst into RUN_CNT single lists
    # ror_lst_lst = [item for sublist in ror_lst_lst for item in sublist]
    # print(f"Length of ror_lst_lst: {len(ror_lst_lst)}")
    # print(f"Length of ror_lst_lst[0]: {len(ror_lst_lst[0])}")
    # ror_df = pd.DataFrame(ror_lst_lst, index = ASSET_CLASSES.keys())
    ror_df = ror_df.apply(lambda x: (x-1) * 100)
    print(f"ROR DF shape:\n{ror_df.shape}")
    ror_mean = ror_df.mean(axis=1)
    ror_std = ror_df.std(axis=1)
    print(f"ROR Mean:\n{ror_mean}")
    print(f"ROR Std:\n{ror_std}")
    # print(f"ROR DF:\n{ror_df}")
    results_df = stats_df.copy(deep=True)
    # Add the mean and std of the ror_df to the results_df mak
    # reindex ror_mean and ror_std to match the index of stats_df
    ror_mean = ror_mean.reindex(stats_df.index)
    ror_std = ror_std.reindex(stats_df.index)
    results_df['Mean Computed'] = ror_mean
    results_df['Std Computed'] = ror_std
    results_df['Mean Delta'] = results_df['Mean'] - results_df['Mean Computed']
    results_df['Std Delta'] = results_df['Std'] - results_df['Std Computed']
    print(f"Results DF:\n{results_df}")

    # compute the cross-correlation of the rows of ror_df
    corr_matrix = np.corrcoef(ror_df, rowvar=True)
    corr_df = pd.DataFrame(corr_matrix, index=ASSET_CLASSES.keys(), columns=ASSET_CLASSES.keys())
    print(f"Corr Matrix:\n{corr_df}")


    # Validate that we get the same ror_df whether we have nb_ages and run_cnt equal to (10, 1000), (100, 100), (1000, 10)
    print("-----------------")
    print(f"Validating that we get the same ror_df whether we have nb_ages and run_cnt equal to (10, 1000), (100, 100), (1000, 10)\n")
    test_cases = [ (10, 1000), (100, 100), (1000, 10) ]
    results_df = stats_df.copy(deep=True)
    for test_nb in range(len(test_cases)):
        nb_ages, run_cnt = test_cases[test_nb]
        print(f"Running test {test_nb} with nb_ages: {nb_ages} and run_cnt: {run_cnt}")
        config_manager.config['nb_ages'] = nb_ages
        config_manager.config['run_cnt'] = run_cnt
        ror_df = pd.DataFrame(columns=range(run_cnt*nb_ages))  # need to specify the number of columns

        for nb, gener in enumerate(ror_gen_list):
            name = gener.name  # get the asset class name from the generator
            for _ in range(RUN_CNT):
                # generator returns a list which contains the list of values for each age
                new_ror_values = list(next(gener))
                ror_lst_lst[nb].extend(new_ror_values) 
            ror_df.loc[name] = pd.Series(ror_lst_lst[nb])
        ror_df = ror_df.apply(lambda x: (x-1) * 100)
        ror_mean = ror_df.mean(axis=1)
        ror_std = ror_df.std(axis=1)
        ror_mean = ror_mean.reindex(stats_df.index)
        ror_std = ror_std.reindex(stats_df.index)
        results_df[f'Mean Computed-{test_nb}'] = ror_mean
        results_df[f'Std Computed-{test_nb}'] = ror_std
    print(f"Validation Results DF:\n{results_df}")

    # Validate that if we use 4 CPUs, the 4 RoR dfs are different
    print("-----------------")
    print(f"Validating that if we use 4 CPUs, the 4 RoR dfs are different\n")
    nb_cpu = 4
    ror_gen_list = []
    for ror_tuple, name, seed in zip(ROR_STATS, ASSET_CLASSES.keys(), seed_sequence):
        array_rand_gen = ArrayRandGen(config_manager, name, ror_tuple[0], ror_tuple[1], seed)
        ror_gen_list.append(array_rand_gen)
    ror_df_list = []
    for gener in ror_gen_list:


    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
