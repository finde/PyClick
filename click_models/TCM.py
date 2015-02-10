#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper, ClickModel, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY
import sys
import numpy as np
import math


from functools import partial
from collections import defaultdict

MAX_ITERATIONS = 40
PRETTY_LOG = True

__author__ = 'Ilya Markov'


class TCM(ClickModel):
    def init_params(self, init_param_values):
        self.param_helper = TCMParamHelper()
        params = {
            TCMExamination.NAME: TCMExaminationWrapper(init_param_values, TCMExamination.default(),
                                                       param_helper=self.param_helper),
            TCMRelevance.NAME: TCMRelevanceWrapper(init_param_values, TCMRelevance.default(),
                                                   param_helper=self.param_helper),
            TCMIntent.NAME: TCMIntentWrapper(init_param_values, TCMIntent.default(), param_helper=self.param_helper),
            TCMAnother.NAME: TCMAnotherWrapper(init_param_values, TCMAnother.default(), param_helper=self.param_helper),
            TCMFreshness.NAME: TCMFreshnessWrapper(init_param_values, TCMFreshness.default(),
                                                   param_helper=self.param_helper)
        }
        return params

    def train(self, tasks):
        """
            Trains the model.
        """
        if len(tasks) <= 0:
            print >>sys.stderr, "The number of training sessions is zero."
            return

        self.params = self.init_params(self.get_prior_values())

        self.param_helper.calculate_intent(self.params, tasks)
        
        for iteration_count in xrange(MAX_ITERATIONS):
            self.params = self.get_updated_params(tasks, self.params)

            if not PRETTY_LOG:
                print >>sys.stderr, 'Iteration: %d, LL: %.10f' % (iteration_count + 1, self.get_loglikelihood(sessions))

        print self.params[TCMIntent.NAME]


    def get_updated_params(self, tasks, priors):
        updated_params = priors

        for task in tasks:
            for session in task:
                for rank, result in enumerate(session.web_results):
                    params = self.get_params(self.params, session, rank)
                    param_values = self.get_param_values(params)
                    current_params = self.get_params(updated_params, session, rank)
                    self.update_param_values(current_params, param_values, session, rank)
        return updated_params

    def get_freshness(self, task):
    
        beta = self.params[TCMExamination.NAME]
        docs = {}
        position = {}
        # p_already_shown + p_is_new
        for query_index, query in enumerate(task):
            for rank, result in enumerate(query.web_results):
                if not position.has_key(result.object):
                    position[result.object] = [None] * len(task)
                    docs[result.object] = [0] * len(task)

                # store rank for proba calculation
                position[result.object][query_index] = rank

        # calculate probability of if it already shown and examined
        # if it the first query, everything is fresh
        for d, queries in position.items():
            last_prob = 0
            for j, v in enumerate(queries):
                if v is not None:
                    # will reach here if the document already shown before
                    docs[d][j] = last_prob * beta.get_param(v)
                last_prob = docs[d][j]
        
        return docs


    def get_p_click(self, param_values):
        beta = param_values[TCMExamination.NAME]
        alpha_1 = param_values[TCMIntent.NAME]
        alpha_2 = param_values[TCMAnother.NAME]
        alpha_3 = param_values[TCMFreshness.NAME]
        r_d = param_values[TCMRelevance.NAME]

        p_click = alpha_1 * beta * r_d * alpha_3
        return p_click


    def get_loglikelihood(self, tasks):
        loglikelihood = 0
        total_sessions = 0
        for task in tasks:
            for session in task:
                log_click_probs = self.get_log_click_probs(session)
                loglikelihood += sum(log_click_probs) / len(log_click_probs)
            total_sessions += len(task)
        loglikelihood /= total_sessions
        return loglikelihood

    def get_perplexity(self, tasks):
        """
        Returns the perplexity and position perplexities for given sessions.
        """
        perplexity_at_rank = [0.0] * MAX_DOCS_PER_QUERY
        total_session = 0
        for task in tasks:
            for session in task:
                log_click_probs = self.get_log_click_probs(session)
                for rank, log_click_prob in enumerate(log_click_probs):
                    perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)
            total_session += len(task)
        perplexity_at_rank = [2 ** (-x / total_session) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)

        return perplexity, perplexity_at_rank

    def predict_click_probs(self, session):  
        #NOTE(Luka): Don't think this is necessary for testing purposes
        #NOTE(Luka): Might become task instead of session
        #TODO(Luka): This is wrong as we have no session but tasks in code. Need to change back
        ''.bla
        beta = self.params[TCMExamination.NAME]
        alpha_1 = self.params[TCMIntent.NAME]
        alpha_2 = self.params[TCMAnother.NAME]
        alpha_3 = self.params[TCMFreshness.NAME]
        r_d = self.params[TCMRelevance.NAME]

        p_click = 0
        for task in tasks:
            freshness = self.get_freshness(task)
            for session in tasks:
                # for each result
                for q, query in enumerate(session):
                    p_match_user_intention = alpha_1.get_param(query).get_value()
                    
                    # for each document
                    # j = document rank for current query
                    # d = document object
                    for j, d in enumerate(query.web_results):

                        p_examine_document = beta.get_param(j).get_value()
                        
                        p_relevance = r_d.get_param(query, d.object).get_value()
                        
                        p_fresh = alpha_3.get_param(query,d.object).get_value()
                        p_fresh *= freshness[d.object][q_index]
                        p_fresh += 1 - freshness[d.object][q_index]

                        p_click += np.log10(p_match_user_intention * p_examine_document * p_relevance * p_fresh)




    def from_JSON(self, json_str):
        param_helper_backup = self.param_helper
        super(UBM, self).from_JSON(json_str)
        self.param_helper = param_helper_backup


    def __str__(self):
        s = self.__class__.__name__ + "\nParams:"
        for p in self.params.keys():
            s += "\n\t" + str(p)
        return s

    @staticmethod
    def get_prior_values():
        return {
            TCMRelevance.NAME: 0.5,
            TCMExamination.NAME: [1 for i in xrange(MAX_DOCS_PER_QUERY)],
            TCMIntent.NAME: 1,
            TCMAnother.NAME: 0.5,
            TCMFreshness.NAME: 0.5,
        }


class TCMParamHelper(object):
    
    def calculate_intent(self,params,tasks):
        for task in tasks:
            for session in task:
                intent = params[TCMIntent.NAME].get_param(session,0)
                if any(session.get_clicks()):
                    intent.numerator += 1
                intent.denominator += 1
    
class TCMRelevance(ClickModelParam):
    """
        Probability of relevance: rel = P(R_ij = 1) = r_d.
    """

    #NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)

    NAME = "rel"

    def update_value(self, param_values, click):
        pass        


class TCMExamination(ClickModelParam):
    """
        Examination probability: exam = P(E_ij = 1). beta_j
    """
    
    #NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)

    NAME = "exam"

    def update_value(self, param_values, click):
        pass


class TCMIntent(ClickModelParam):
    """
        Matches user intent probability: intent = P(M_i = 1). alpha1
    """

    #NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)
    
    NAME = "intent"

    def update_value(self, param_values, click):
        pass


class TCMAnother(ClickModelParam):
    """
        Whether user submits other query after i: another = P(N_i = 1 | M_i = 1). alpha2
    """

    #NOTE(Luka): Only depends on Intent (M_i)
    
    NAME = "another"

    def update_value(self, param_values, click):
        pass


class TCMFreshness(ClickModelParam):
    """
        Freshness of document: fresh = P(F_ij | H_ij). alpha3
        H_ij = Previous examination of doc at rank j in session (Binary)

    """

    #NOTE(Luka): Depends on Previous examination (H_ij)
    
    NAME = "fresh"

    def update_value(self, param_values, click):
        pass


class TCMExaminationWrapper(ClickModelParamWrapper):

    # Shape: (max_rank,1)
    # Max rank in Yandex is 10

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        
        return [ init_param_func(i, **kwargs) for i in xrange(MAX_DOCS_PER_QUERY)]

    def init_param_function(self, rank = 0, **kwargs):
        self.init_value = self.init_param_values[self.name][rank]
        if self.init_value >= 0:
            param = self.factory.init(self.init_value, **kwargs)
        else:
            param = self.factory.default(**kwargs)

        return param

    
    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given rank.
        """
        return self.params[rank-1]

    def get_params_from_JSON(self, json_str):
        pass


class TCMRelevanceWrapper(ClickModelParamWrapper):
    # Shape: (N_queries,N_docs) 
    # N_docs that are on all result pages for query, so prob defaultdict of a defaultdict
    # with an initialized value?

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        init = partial(init_param_func,**kwargs)
        params = defaultdict(lambda : defaultdict(init))
        return params

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query and doc.
        """
        return self.params[session.query][rank]

    def get_params_from_JSON(self, json_str):
        pass


class TCMIntentWrapper(ClickModelParamWrapper):
    # Shape: (N_query,1)

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return defaultdict(lambda : init_param_func(**kwargs))


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query.
        """
        return self.params[session.query]
    
    def get_params_from_JSON(self, json_str):
        pass


class TCMAnotherWrapper(ClickModelParamWrapper):

    # Shape: (Sessions, 1)

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return defaultdict(lambda : init_param_func(**kwargs))


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query.
        """
        return self.params[session.query]
    
    def get_params_from_JSON(self, json_str):
        pass


class TCMFreshnessWrapper(ClickModelParamWrapper):
    # Shape: (N_queries,N_docs)
    # See TCMRelevance for explanation

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        init = partial(init_param_func,**kwargs)
        params = defaultdict(lambda : defaultdict(init))
        return params

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query and document.
        """
        return self.params[session.query][rank]

    def get_params_from_JSON(self, json_str):
        pass



