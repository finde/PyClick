#!/usr/bin/env python
#
# Copyright (C) 2014 Luka Stout and Finde Xumara
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

__author__ = 'Luka Stout and Finde Xumara'


def main():
    this_directory = os.path.dirname(os.path.realpath(__file__))

    sessions_dict = parse_yandex_sessions(os.path.join(this_directory, 'data', 'YandexClicks-sample.txt'), 10000)
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
        print("--- %s seconds ---" % (time.time() - start_time))

        print click_model

        print "Log-likelihood and perplexity"
        log_likelihood, perplexity, perplexity_at_rank = click_model.test(_test)
        print log_likelihood, perplexity
        print ""


if __name__ == '__main__':
    main()
