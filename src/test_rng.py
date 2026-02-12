#!/usr/local/bin/python3
"""
Tests for the random number generator
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from arrayrandgen_class import ArrayRandGen
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


class TestRng:
    """
    Tests for the random number generator
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        """
    
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.start_age = self.config['start_age']
        self.end_age = self.config['end_age']  # number of ages in the range
        self.nb_ages = self.end_age - self.start_age + 1
        self.stats_df = pd.DataFrame(ROR_STATS, index=ASSET_CLASSES.keys(), columns=['Mean', 'Std'])
        self.nb_assets = len(ASSET_CLASSES.keys())
        self.stats_lst = [(name, mean, stdv) for name, mean, stdv in zip(self.stats_df.index, self.stats_df['Mean'], self.stats_df['Std'])]
        logger.info(f"Stats DF:\n{self.stats_df}")
        # Create a sequence of pseudo-random seeds for the random number generators for each CPU
        master_seed_sequence = np.random.SeedSequence(MASTER_SEED)
        self.seed_sequence = master_seed_sequence.spawn(NB_CPU)

        return None


    def run_test_1(self) -> pd.DataFrame:
        self.run_cnt = RUN_CNT
        ror_gen_list = []
        for (name, mean, stdv), seed in zip(self.stats_lst, self.seed_sequence):
            array_rand_gen = ArrayRandGen(self.config_manager, name, mean, stdv, seed)
            ror_gen_list.append(array_rand_gen)
        ror_lst_lst = [[] for _ in range(self.nb_assets)]
        ror_df = pd.DataFrame(columns=range(self.run_cnt*self.nb_ages))  # need to specify the number of columns
        for nb, gener in enumerate(ror_gen_list):
            name = gener.name  # get the asset class name from the generator
            for _ in range(self.run_cnt):
                # generator returns a list which contains the list of values for each age
                new_ror_values = list(next(gener))
                ror_lst_lst[nb].extend(new_ror_values) 
            ror_df.loc[name] = pd.Series(ror_lst_lst[nb])
        ror_df = ror_df.apply(lambda x: (x-1) * 100)
        logger.info(f"ROR DF shape:\n{ror_df.shape}")
        ror_mean = ror_df.mean(axis=1)
        ror_std = ror_df.std(axis=1)
        logger.info(f"ROR Mean:\n{ror_mean}")
        logger.info(f"ROR Std:\n{ror_std}")
        # print(f"ROR DF:\n{ror_df}")
        results_df = self.stats_df.copy(deep=True)
        # Add the mean and std of the ror_df to the results_df mak
        # reindex ror_mean and ror_std to match the index of stats_df
        ror_mean = ror_mean.reindex(self.stats_df.index)
        ror_std = ror_std.reindex(self.stats_df.index)
        results_df['Mean Computed'] = ror_mean
        results_df['Std Computed'] = ror_std
        results_df['Mean Delta'] = results_df['Mean'] - results_df['Mean Computed']
        results_df['Std Delta'] = results_df['Std'] - results_df['Std Computed']
        # print(f"Results DF:\n{results_df}")
        return results_df



def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    test_rng = TestRng(config_manager)
    results_df = test_rng.run_test_1()
    logger.info(f"Results DF:\n{results_df}")

    return None


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
    print(f"Validating that if we use {NB_CPU} CPUs, the {NB_CPU} RoR dfs are different\n")
    run_cnt = 1000
    nb_ages = 35
    config_manager.config['nb_ages'] = nb_ages
    config_manager.config['run_cnt'] = run_cnt
    print(f"NB_CPU: {NB_CPU} - nb_ages: {nb_ages} - run_cnt: {run_cnt}")
    # Create a sequence of pseudo-random seeds for the random number generators for each CPU
    master_seed_sequence = np.random.SeedSequence(MASTER_SEED)
    seed_sequence = master_seed_sequence.spawn(NB_CPU)
    print(f"Seed sequence:\n{seed_sequence}")
    print(f"the ror_df should be different for each seed, but the mean and std should be the same, but not exactly")
    print(f"... and equal to the values of stats_df (first 2 columns)")
    results_df = stats_df.copy(deep=True)
    for seed_nb, seed in enumerate(seed_sequence):
        ror_df = pd.DataFrame(columns=range(run_cnt*nb_ages))  # need to specify the number of columns
        for name, mean, stdv in stats_lst:
            array_rand_gen = ArrayRandGen(config_manager, name, mean, stdv, seed)
            ror_df.loc[name] = pd.Series(list(next(array_rand_gen)))
        ror_df = ror_df.apply(lambda x: (x-1) * 100)
        # print the first 12 columns of the ror_df
        print(f"ror_df shape: {ror_df.shape}")
        print(f"First 12 columns of ROR DF for seed {seed_nb}:\n{ror_df.iloc[:, :12]}")
        ror_mean = ror_df.mean(axis=1)
        ror_std = ror_df.std(axis=1)
        ror_mean = ror_mean.reindex(stats_df.index)
        ror_std = ror_std.reindex(stats_df.index)
        results_df[f'Mean Computed-{seed_nb}'] = ror_mean
        results_df[f'Std Computed-{seed_nb}'] = ror_std
    print(f"Validation Results DF:\n{results_df}")

    # Calling one set of ArrayRandGen - NB_CPU times - to generate the RoR series - initialized with one seed should produce
    # results with similar statistics 
    print("-----------------")
    run_cnt = 1000
    nb_ages = 35
    config_manager.config['nb_ages'] = nb_ages
    config_manager.config['run_cnt'] = run_cnt
    print(f"nb_ages: {nb_ages} - run_cnt: {run_cnt}")
    print(f"Calling one set of ArrayRandGen - nb_cpu times - to generate the RoR series - initialized with one seed should produce")
    print(f"results with similar statistics\n")
    results_df = stats_df.copy(deep=True)
    # Create the list of generators - once and for all for this test
    ror_gen_list = [ArrayRandGen(config_manager, name, mean, stdv, MASTER_SEED) for name, mean, stdv in stats_lst]

    # Call the generators NB_CPU times - to generate NB_CPU RoR series
    for cpu_nb in range(10):
        ror_df = pd.DataFrame(columns=range(run_cnt*nb_ages))  # need to specify the number of columns
        for gener in ror_gen_list:
            name = gener.name  # get the asset class name from the generator
            ror_df.loc[name] = pd.Series(list(next(gener)))
        ror_df = ror_df.apply(lambda x: (x-1) * 100)
        # print the first 12 columns of the ror_df
        print(f"ror_df shape: {ror_df.shape}")
        print(f"First 12 columns of ROR DF for seed {cpu_nb}:\n{ror_df.iloc[:, :12]}")
        ror_mean = ror_df.mean(axis=1)
        ror_std = ror_df.std(axis=1)
        ror_mean = ror_mean.reindex(stats_df.index)
        ror_std = ror_std.reindex(stats_df.index)
        results_df[f'Mean Computed-{cpu_nb}'] = ror_mean
        results_df[f'Std Computed-{cpu_nb}'] = ror_std
    print(f"Validation Results DF:\n{results_df}")



    test_rng.run()
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
