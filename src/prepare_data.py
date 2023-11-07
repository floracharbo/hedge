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

import torch

from data_preparation.clustering import Clusterer
from data_preparation.importer import import_data
from data_preparation.parameters import get_parameters
from data_preparation.scaling_factors import scaling_factors

if __name__ == "__main__":
    # 0 - initialise variables
    tic_dtm = datetime.datetime.now()
    prm = get_parameters()
    dtm_1 = datetime.datetime.now()

    print(f"torch.cuda.is_available() {torch.cuda.is_available()}")

    # 1 - import generation and electricity demand data
    print("(1) import profiles")
    days, n_data_type = import_data(prm)
    dtm_2 = datetime.datetime.now()
    print(f"(1) done profiles import in {(dtm_2 - dtm_1)/60} minutes")

    # 2 - clustering - for demand and transport
    print("(2) clustering")
    clusterer = Clusterer(prm)
    banks = clusterer.clustering(days, n_data_type)
    dtm_3 = datetime.datetime.now()
    print(f"(2) done clustering in {(dtm_3 - dtm_2)/60} minutes")

    # 3 - scaling factors
    print("(3) scaling factors")
    scaling_factors(prm, banks, days, n_data_type)
    dtm_4 = datetime.datetime.now()
    print(f"(3) done scaling factors in {(dtm_4 - dtm_3)/60} minutes")

    toc_dtm = datetime.datetime.now()
    print(f"END. Total duration {(toc_dtm - tic_dtm).seconds/60} minutes")
