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

from evaluation.Evaluation import *

import time
from reader import parse_yandex_sessions, parse_yandex_relevances

from sklearn.cross_validation import train_test_split
from tabulate import tabulate

__author__ = 'Luka Stout and Finde Xumara'


def main(sessions_file, relevance_file, n_sessions):
    this_directory = os.path.dirname(os.path.realpath(__file__))

    sessions_dict = parse_yandex_sessions(os.path.join(this_directory, sessions_file), int(n_sessions))

    train, test = train_test_split(sessions_dict)

    classes = [
        UBM,
        TCM,
        SimpleDCM,
        SimpleDBN,
        DCM,
        DBN,
        #FCM,
        #VCM
    ]

    headers = ['Model', 'LL', 'Perp', 'Rel.Pred.MSE', 'Ranking.NDCG', 'CTR Pred.', 'Comp. Time.']
    tableData = []
    true_relevances = parse_yandex_relevances(os.path.join(this_directory, relevance_file))  
    rel_pred = RelevancePrediction(true_relevances)
    ll = Loglikelihood()
    perp = Perplexity()
    ranking = RankingPerformance(true_relevances)

    for click_model_class in classes:

        print "==== %s ====" % click_model_class.__name__
        click_model = click_model_class(click_model_class.get_prior_values())

        start_time = time.time()
        click_model.train(train)
        training_time = (time.time() - start_time)
        print("--- %s seconds ---" % training_time)

        #print click_model

        
        print "Log-likelihood"  
        log_likelihood = ll.evaluate(click_model, test)
        print log_likelihood
        
        print "Perplexity"
        perplexity, perplexity_at_rank = perp.evaluate(click_model, test)
        print perplexity, 
        print " ".join(["%.4f" % v for v in perplexity_at_rank])
        
        print "Relevance prediction MSE"
        rel_score = rel_pred.evaluate(click_model, test)
        print rel_score

        print "Relevance Ranking"
        rank_score = ranking.evaluate(click_model, train+test)
        print rank_score
        
        print 'Click through rate prediction'
        print 'Not yet implemented'
        ctr_score = 0

        print ""

        tableData.append([click_model_class.__name__, log_likelihood, perplexity, rel_score, rank_score, ctr_score, training_time])

    _format = 'grid'
    print '\n\nSUMMARY\n=========='
    print 'Number of queries: ', n_sessions
    print tabulate(tableData, headers, tablefmt=_format)


if __name__ == '__main__':
    session_data = sys.argv[1]
    relevance_data = sys.argv[2]
    if len(sys.argv) >= 4:
        sessions = sys.argv[3]
    else:
        sessions = 100
    main(session_data, relevance_data, sessions)
