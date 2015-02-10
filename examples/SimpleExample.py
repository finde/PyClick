#!/usr/bin/env python
#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
from click_models.DBN import DBN
from click_models.DCM import DCM
from click_models.SimpleDBN import SimpleDBN
from click_models.SimpleDCM import SimpleDCM
from click_models.UBM import UBM
from click_models.TCM import TCM
from reader import parse_yandex_sessions, transform_to_tasks
from session.Session import *

__author__ = 'Ilya Markov'


def parse_wsdm_sessions(sessions_filename):
    """Parses search sessions in the given file into Session objects."""

    sessions_file = open(sessions_filename, "r")
    sessions = []

    for line in sessions_file:
        for session_str in line.split(";"):
            if session_str.strip() == "":
                continue

            session_str = session_str.strip()
            session_str = session_str.split("\t")
            query = session_str[0]

            session_str = session_str[1].split(":")
            docs = session_str[0].strip().split(",")
            clicks = session_str[1].strip().split(",")

            session = Session(query)
            for doc in docs:
                web_result = Result(doc, -1, 1 if doc in clicks else 0)
                session.add_web_result(web_result)

            sessions.append(session)

    return sessions


def main(train_filename):

    sessions_dict = parse_yandex_sessions(train_filename, 20)

    classes = [
        TCM,
        UBM,
        # SimpleDCM,
        # SimpleDBN,
        # DBN,
        # UBM
    ]

    # TODO: fix initialization
    for click_model_class in classes:

        if click_model_class.__name__ == TCM.__name__:
            sessions = transform_to_tasks(sessions_dict)
        else:
            sessions = [s for t in sessions_dict.values() for s in t]

        print "==== %s ====" % click_model_class.__name__
        click_model = click_model_class(click_model_class.get_prior_values())
        click_model.train(sessions)

        print click_model

        print "Log-likelihood and perplexity"
        print click_model.test(sessions)
        print ""


# An example of using PyClick.
if __name__ == '__main__':
    main('data/YandexClicks-sample.txt')

    # if len(sys.argv) < 2:
    # print "USAGE: %s <file with train sessions> <file with test sessions>" % sys.argv[0]
    #     sys.exit(1)
    #
    # main(sys.argv[1])
