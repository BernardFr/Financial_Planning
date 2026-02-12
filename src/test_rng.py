#!/usr/local/bin/python3
"""
Tests for the random number generator
"""


import numpy as np
import pandas as pd
from multiprocessing import Pool

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
RUN_CNT_LARGE = 10_000  # used for large runs to get a good estimate of the mean and std of the random numbers
AGE_RUN_CNT_LIST = [ (10, 1000), (100, 100), (1000, 10) ]
TESTS_TO_RUN = {'basic': False, 'age_run_cnt': False, 'multiple_cpus': True}


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


    def run_test_basic(self, run_cnt: int = RUN_CNT_LARGE) -> pd.DataFrame:
        """ 
        Basic test of the random number generator
        Creates a list of random number generators for each asset class
        Creates a list of lists of random numbers for each asset class
        Creates a DataFrame of the random numbers
        Computes the mean and std of the random numbers
        Returns the results as a DataFrame

        Note: self.run_cnt should be set a large value to get a good estimate of the mean and std of the random numbers
        """
        self.run_cnt = run_cnt
        logger.info(f"Running test with run_cnt: {self.run_cnt:,d}")
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
        # reindex ror_mean and ror_std to match the index of stats_df
        ror_mean = ror_mean.reindex(self.stats_df.index)
        ror_std = ror_std.reindex(self.stats_df.index)
        results_df['Mean Computed'] = ror_mean
        results_df['Std Computed'] = ror_std
        results_df['Mean Delta'] = results_df['Mean'] - results_df['Mean Computed']
        results_df['Std Delta'] = results_df['Std'] - results_df['Std Computed']
        return results_df

    def run_test_age_run_cnt(self) -> pd.DataFrame:
    # Validate that we get the same ror_df whether we have nb_ages and run_cnt equal to (10, 1000), (100, 100), (1000, 10)
        test_cases = AGE_RUN_CNT_LIST
        logger.info(f"Validating that we get the same ror_df with different nb_ages and run_cnt and nb_ages * run_cnt constant")
        logger.info(f"test_cases: {test_cases}")
        results_df = self.stats_df.copy(deep=True)
        # generate differnt permutations of nb_ages and run_cnt - nb_ages * run_cnt should be constant
        for test_nb, (nb_ages, run_cnt) in enumerate(test_cases):
            logger.info(f"Running test {test_nb} with nb_ages: {nb_ages} and run_cnt: {run_cnt}")
            self.config_manager.config['nb_ages'] = nb_ages
            self.config_manager.config['run_cnt'] = run_cnt
            ror_df = pd.DataFrame(columns=range(run_cnt*nb_ages))  # need to specify the number of columns

            #Re-create a list of random number generators for each asset class, i.e. to reset the seed to same value
            ror_gen_list = []
            for (name, mean, stdv), seed in zip(self.stats_lst, self.seed_sequence):
                array_rand_gen = ArrayRandGen(self.config_manager, name, mean, stdv, seed)
                ror_gen_list.append(array_rand_gen)

            # Foreach asset classe we generate nb_ages values - run_cnt times into the list
            for gener in ror_gen_list: # iterate of each asset class generator
                name = gener.name  # get the asset class name from the generator
                ror_lst = []
                for _ in range(self.run_cnt):
                    # generator returns a list which contains the list of values for each age
                    new_ror_values = list(next(gener))
                    ror_lst.extend(new_ror_values) 
                ror_df.loc[name] = pd.Series(ror_lst)
            ror_df = ror_df.apply(lambda x: (x-1) * 100)
            ror_mean = ror_df.mean(axis=1)
            ror_std = ror_df.std(axis=1)
            ror_mean = ror_mean.reindex(self.stats_df.index)
            ror_std = ror_std.reindex(self.stats_df.index)
            results_df[f'Mean Computed-{test_nb}'] = ror_mean
            results_df[f'Std Computed-{test_nb}'] = ror_std
        return results_df

    def run_test_cpu(self, seed_sequence: np.random.SeedSequence, run_cnt: int) -> pd.DataFrame:
        """ 
        Returns a DF of RoR
        Rows are asset classes, columns are ages - repeated run_cnt times
        Note that seed_sequence is one SeedSequence per CPU; spawn child seeds for each asset class.
        """
        ror_gen_list = []
        # Spawn a child seed for each asset class from this CPU's seed sequence
        asset_seeds = seed_sequence.spawn(self.nb_assets)
        for (name, mean, stdv), seed in zip(self.stats_lst, asset_seeds):
            array_rand_gen = ArrayRandGen(self.config_manager, name, mean, stdv, seed)
            ror_gen_list.append(array_rand_gen)
        ror_df = pd.DataFrame(columns=range(run_cnt*self.nb_ages))  # need to specify the number of columns
        for gener in ror_gen_list:
            name = gener.name  # get the asset class name from the generator
            ror_lst = []
            for _ in range(self.run_cnt):
                # generator returns a list of ror values for each age - total nb_ages values
                new_ror_values = list(next(gener))
                ror_lst.extend(new_ror_values) # ror_lst has nb_ages * run_cnt values
            ror_df.loc[name] = pd.Series(ror_lst)
        ror_df = ror_df.apply(lambda x: (x-1) * 100)
        return ror_df

    def run_test_multiple_cpus(self, run_cnt: int = RUN_CNT_LARGE) -> pd.DataFrame:
        """
        Validate that if we use 4 CPUs, the 4 RoR dfs are different, but have similar statistics and similar to the target stats_df
        Also validate that combining the 4 RoR dfs into one large RoR df gives the same statistics as generating the RoR df with one CPU

        ChatGPT:
        def worker(child_seedseq, n):
            rng = np.random.default_rng(child_seedseq)
            return rng.normal(size=n).mean()

        if __name__ == "__main__":
            root = np.random.SeedSequence(12345)
            child_seqs = root.spawn(8)  # one per process/task

            with Pool(8) as p:
                results = p.starmap(worker, [(child_seqs[i], 1_000_000) for i in range(8)])

            print(results)
        """
        logger.info(f"Validating that if we use {NB_CPU} CPUs, the {NB_CPU} RoR dfs are different\n")
        run_cnt_cpu = int(run_cnt / NB_CPU)
        self.run_cnt = run_cnt_cpu * NB_CPU  # Total number of simulations to run (in case of rounding errors)
        logger.info(f"NB_CPU: {NB_CPU} - nb_ages: {self.nb_ages} - run_cnt: {self.run_cnt:,d} - run_cnt_cpu: {run_cnt_cpu:,d}")
        # Create a sequence of pseudo-random seeds for the random number generators for each CPU
        master_seed_sequence = np.random.SeedSequence(MASTER_SEED)
        seed_sequence = master_seed_sequence.spawn(NB_CPU)
        param_list = [(seed_seq, run_cnt_cpu) for seed_seq in seed_sequence]
        with Pool(NB_CPU) as p:
            # ror_df_list = p.starmap(self.run_test_cpu, [(seed, run_cnt_cpu) for seed in seed_sequence])
            ror_df_list = p.starmap(self.run_test_cpu, param_list)
        # Concatenate the results into a single DataFrame
        for df_nb, df in enumerate(ror_df_list):
            tmp_df = pd.concat([df.mean(axis=1), df.std(axis=1)], axis=1)
            logger.info(f"df_nb: {df_nb} - tmp_df:\n{tmp_df}")
        results_ror_df = pd.concat(ror_df_list,axis=1)
        logger.info(f"results_ror_df shape: {results_ror_df.shape}")
        results_df = self.stats_df.copy(deep=True)
        results_df['Mean Computed'] = results_ror_df.mean(axis=1)
        results_df['Std Computed'] = results_ror_df.std(axis=1)
        results_df['Mean Delta'] = results_df['Mean'] - results_df['Mean Computed']
        results_df['Std Delta'] = results_df['Std'] - results_df['Std Computed']
        return results_df



def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    test_rng = TestRng(config_manager)
    if TESTS_TO_RUN['basic']:
        results_df = test_rng.run_test_basic(run_cnt=RUN_CNT_LARGE)
        logger.info(f"test_run_basic: the computed mean and std should be close to the target Mean and Std")
        logger.info(f"Results DF:\n{results_df}")

    if TESTS_TO_RUN['age_run_cnt']:
        results_df = test_rng.run_test_age_run_cnt()
        logger.info(f"test_run_age_run_cnt  the computed mean and std should be identical")
        logger.info(f"test_run_age_run_cnt  Results DF:\n{results_df}")
    
    if TESTS_TO_RUN['multiple_cpus']:   
        results_df = test_rng.run_test_multiple_cpus(run_cnt=RUN_CNT_LARGE)
        logger.info(f"test_run_multiple_cpus  the computed mean and std should be the same as the target mean and std")
        logger.info(f"test_run_multiple_cpus  Results DF:\n{results_df}")

    return None


if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
