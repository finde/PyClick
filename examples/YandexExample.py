#!/usr/bin/env python
#
# Copyright (C) 2015 Luka Stout and Finde Xumara
#
# Full copyright notice can be found in LICENSE.
#
import sys
import os

from click_models.DBN import DBN
from click_models.DCM import DCM
from click_models.SimpleDBN import SimpleDBN
from click_models.SimpleDCM import SimpleDCM
from click_models.UBM import UBM
from click_models.TCM import TCM
from click_models.FCM import FCM
from click_models.VCM import VCM
from session.Session import *

import time
from reader import parse_yandex_sessions

from sklearn.cross_validation import train_test_split
from tabulate import tabulate

__author__ = 'Luka Stout and Finde Xumara'


def main(data_file, n_sessions):
    this_directory = os.path.dirname(os.path.realpath(__file__))

    sessions_dict = parse_yandex_sessions(os.path.join(this_directory, data_file), int(n_sessions))

    train, test = train_test_split(sessions_dict.values())

    classes = [
        TCM,
        UBM,
        DCM,
        DBN,
        SimpleDCM,
        SimpleDBN,
        # FCM,
        # VCM
    ]

    headers = ['Click Model', 'Log-likelihood', 'Perplexity', 'Computation Time']
    tableData = []
    for click_model_class in classes:

        if not click_model_class.__name__ == TCM.__name__:
            _train = [s for t in train for s in t]
            _test = [s for t in test for s in t]
        else:
            _train = train
            _test = test

        print "==== %s ====" % click_model_class.__name__
        click_model = click_model_class(click_model_class.get_prior_values())

        start_time = time.time()
        click_model.train(_train)
        training_time = (time.time() - start_time)
        print("--- %s seconds ---" % training_time)

        print click_model

        print "Log-likelihood and perplexity"
        log_likelihood, perplexity, perplexity_at_rank = click_model.test(_test)
        print log_likelihood, perplexity
        print ""

        tableData.append([click_model_class.__name__, log_likelihood, perplexity, training_time])

    _format = 'grid'
    print '\n\nSUMMARY\n=========='
    print 'Number of queries: ', n_sessions
    print tabulate(tableData, headers, tablefmt=_format)


if __name__ == '__main__':
    data = sys.argv[1]
    if len(sys.argv) == 3:
        sessions = sys.argv[2]
    else:
        sessions = 1000
    main(data, sessions)
