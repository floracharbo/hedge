#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:47:32 2020.

@author: floracharbonnier

The aim of this script is to take
as input the raw customer-led network revolution (CLNR) data
and to output data in a format that is usable for
the reinforcement learning environment
"""

# import packages
import datetime

from clustering import clustering
from importer import import_data
from parameters import get_parameters
from scaling_factors import scaling_factors

if __name__ == "__main__":
    # 1 - initialise variables
    print("(1) initialise variables")
    tic_dtm = datetime.datetime.now()
    prm, paths, run_config = get_parameters()
    dtm_1 = datetime.datetime.now()
    print(f"(1) done in {dtm_1 - tic_dtm} seconds")

    # %% 2 - import generation and electricity demand data
    print("(2) import profiles")
    days, n_data_type = import_data(prm, paths)
    dtm_2 = datetime.datetime.now()
    print(f"(2) done in {dtm_2 - dtm_1} seconds")

    # %% 4 - clustering - for demand and transport
    print("(3) clustering")
    banks = clustering(
        days, prm, paths["save_path"], n_data_type)
    dtm_3 = datetime.datetime.now()
    print(f"(3) done in {dtm_3 - dtm_2} seconds")

    # %% 5 - scaling factors
    print("(4) scaling factors")
    scaling_factors(prm, banks, days, n_data_type,
                    paths["save_path"])
    dtm_4 = datetime.datetime.now()
    print(f"(4) done in {dtm_4 - dtm_3} seconds")

    toc_dtm = datetime.datetime.now()

    print(f"end duration {toc_dtm - tic_dtm}")
