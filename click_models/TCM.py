#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper, ClickModel, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY, MAX_ITERATIONS, PRETTY_LOG
import sys
import math

from functools import partial
from collections import defaultdict

__author__ = 'Luka Stout and Finde Xumara'


class TCM(ClickModel):
    def init_params(self, init_param_values):
        self.param_helper = TCMParamHelper()
        params = {
            TCMExamination.NAME: TCMExaminationWrapper(init_param_values, TCMExamination.default(),
                                                       param_helper=self.param_helper),
            TCMRelevance.NAME: TCMRelevanceWrapper(init_param_values, TCMRelevance.default(),
                                                   param_helper=self.param_helper),
            TCMIntent.NAME: TCMIntentWrapper(init_param_values, TCMIntent.default(), param_helper=self.param_helper),
            TCMFreshness.NAME: TCMFreshnessWrapper(init_param_values, TCMFreshness.default(),
                                                   param_helper=self.param_helper)
        }
        return params

    def train(self, tasks):
        """
            Trains the model.
        """
        if len(tasks) <= 0:
            print >> sys.stderr, "The number of training sessions is zero."
            return

        self.params = self.init_params(self.get_prior_values())

        self.param_helper.set_intent(self.params, tasks)

        for iteration_count in xrange(MAX_ITERATIONS):
            self.params = self.get_updated_params(tasks, self.params)

            if not PRETTY_LOG:
                print >> sys.stderr, 'Iteration: %d, LL: %.10f' % (iteration_count + 1, self.get_loglikelihood(tasks))


    def get_updated_params(self, tasks, priors):
        updated_params = priors

        for task in tasks:
            self.freshness = None
            for i, session in enumerate(task):

                if not hasattr(session, 'freshness'):
                    if self.freshness is None:
                        self.freshness = self.get_freshness(task)
                    session.freshness = self.freshness[i]

                for rank, result in enumerate(session.web_results):
                    params = self.get_params(self.params, session, rank)
                    param_values = self.get_param_values(params)
                    param_values['freshness'] = session.freshness[rank]
                    current_params = self.get_params(updated_params, session, rank)
                    self.update_param_values(current_params, param_values, session, rank)
        return updated_params


    def get_freshness(self, task):

        freshness = dict()
        # queries = [session.query for session in task]
        for session_index, session in enumerate(task):
            freshness[session_index] = []
            for result in session.web_results:
                for session_2 in task[:session_index]:
                    if result.object in [r.object for r in session_2.web_results]:
                        already_shown = True
                        break
                else:
                    already_shown = False

                freshness[session_index].append(already_shown)

        return freshness


    def get_p_click(self, param_values):
        beta = param_values[TCMExamination.NAME]
        alpha_1 = param_values[TCMIntent.NAME]
        alpha_3 = param_values[TCMFreshness.NAME]
        r_d = param_values[TCMRelevance.NAME]

        freshness = param_values['freshness']

        if freshness:
            f_ij = 1
        else:
            f_ij = alpha_3

        p_click = alpha_1 * beta * f_ij * r_d
        return p_click


    def get_log_click_probs(self, session, freshness):
        """
            Returns the list of log-click probabilities for all URLs in the given session.
        """
        log_click_probs = []

        for rank, click in enumerate(session.get_clicks()):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)
            param_values['freshness'] = freshness[rank]
            click_prob = self.get_p_click(param_values)

            if click == 1:
                log_click_probs.append(math.log(click_prob))
            else:
                log_click_probs.append(math.log(1 - click_prob))

        return log_click_probs


    def get_loglikelihood(self, tasks):
        loglikelihood = 0
        total_sessions = 0
        for task in tasks:
            freshness = self.get_freshness(task)
            for s_idx, session in enumerate(task):
                log_click_probs = self.get_log_click_probs(session, freshness[s_idx])
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
            freshness = self.get_freshness(task)
            for s_idx, session in enumerate(task):
                log_click_probs = self.get_log_click_probs(session, freshness[s_idx])
                for rank, log_click_prob in enumerate(log_click_probs):
                    perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)
            total_session += len(task)
        perplexity_at_rank = [2 ** (-x / total_session) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)

        return perplexity, perplexity_at_rank


    def predict_click_probs(self, task):
        """
            Predicts click probabilities for a given session
        """

        beta = self.params[TCMExamination.NAME]
        alpha_1_value = self.params[TCMIntent.NAME].get_param(0, 0).get_value()
        alpha_2 = self.params[TCMAnother.NAME]
        alpha_3 = self.params[TCMFreshness.NAME]
        r_d = self.params[TCMRelevance.NAME]

        freshness = self.get_freshness(task)

        prob_task = []

        for s_idx, session in enumerate(task):
            prob_session = []
            # for each document
            # j = document rank for current query
            # d = document object
            for j, d in enumerate(session.web_results):
                beta_j = beta.get_param(session, j).get_value()
                r_ij = r_d.get_param(session, j).get_value()

                if freshness[s_idx][j]:
                    f_ij = 1
                else:
                    f_ij = alpha_3.get_param(session, j).get_value()

                p_click = alpha_1_value * beta_j * r_ij * f_ij
                prob_session.append(p_click)
            prob_task.append(prob_session)

        return prob_task


    def from_JSON(self, json_str):
        param_helper_backup = self.param_helper
        super(UBM, self).from_JSON(json_str)
        self.param_helper = param_helper_backup


    def __str__(self):
        s = self.__class__.__name__ + "\nParams:"
        s += "\n\t" + TCMRelevance.NAME + " " + str(len(self.params[TCMRelevance.NAME])) + " parameters"
        s += "\n\t" + TCMExamination.NAME + " " + str(self.params[TCMExamination.NAME].params)
        s += "\n\t" + TCMIntent.NAME + " " + str(self.params[TCMIntent.NAME].get_param(0, 0).get_value())
        s += "\n\t" + TCMFreshness.NAME + " " + str(self.params[TCMFreshness.NAME].get_param(0, 0).get_value())
        return s

    @staticmethod
    def get_prior_values():
        return {
            TCMRelevance.NAME: 0.5,
            TCMExamination.NAME: [0.5 for i in xrange(MAX_DOCS_PER_QUERY)],
            TCMIntent.NAME: 0.5,
            TCMFreshness.NAME: 0.5,
        }


class TCMParamHelper(object):
    def set_intent(self, params, tasks):
        intent = params[TCMIntent.NAME].get_param(0, 0)
        for task in tasks:
            for session in task:
                if any(session.get_clicks()):
                    intent.numerator += 1
                intent.denominator += 1


class TCMRelevance(ClickModelParam):
    """
        Probability of relevance: rel = P(R_ij = 1) = r_d.
    """

    # NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)

    NAME = "rel"

    def update_value(self, param_values, click):
        r_ij = param_values[self.NAME]
        alpha_1 = param_values[TCMIntent.NAME]
        alpha_3 = param_values[TCMFreshness.NAME]
        freshness = param_values['freshness']
        beta_j = param_values[TCMExamination.NAME]

        if freshness:
            f_ij = 1
        else:
            f_ij = alpha_3

        if click:
            self.numerator += 1
        else:
            num = r_ij - alpha_1 * beta_j * f_ij * r_ij
            denom = 1 - alpha_1 * beta_j * f_ij * r_ij
            self.numerator += num / denom
        self.denominator += 1


class TCMExamination(ClickModelParam):
    """
        Examination probability: exam = P(E_ij = 1). beta_j
    """

    # NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)

    NAME = "exam"

    def update_value(self, param_values, click):
        r_ij = param_values[TCMRelevance.NAME]
        alpha_1 = param_values[TCMIntent.NAME]
        alpha_3 = param_values[TCMFreshness.NAME]
        freshness = param_values['freshness']
        beta_j = param_values[TCMExamination.NAME]

        if freshness:
            f_ij = 1
        else:
            f_ij = alpha_3

        if click:
            self.numerator += 1
        else:
            num = beta_j - alpha_1 * beta_j * f_ij * r_ij
            denom = 1 - alpha_1 * beta_j * f_ij * r_ij
            self.numerator += num / denom
        self.denominator += 1


class TCMIntent(ClickModelParam):
    """
        Matches user intent probability: intent = P(M_i = 1). alpha1
    """

    # NOTE(Luka): Does not depend on any hidden variables (according to fig. 4)

    NAME = "intent"

    # Intent is set before the EM steps using MLE
    # So is not updated
    def update_value(self, param_values, click):
        pass


class TCMFreshness(ClickModelParam):
    """
        Freshness of document: fresh = P(F_ij | H_ij). alpha3
        H_ij = Previous examination of doc at rank j in session (Binary)

    """

    # NOTE(Luka): Depends on Previous examination (H_ij)

    NAME = "fresh"

    def update_value(self, param_values, click):
        r_ij = param_values[TCMRelevance.NAME]
        alpha_1 = param_values[TCMIntent.NAME]
        alpha_3 = param_values[TCMFreshness.NAME]
        beta_j = param_values[TCMExamination.NAME]

        if click:
            self.numerator += 1
        else:
            num = alpha_3 - alpha_1 * beta_j * alpha_3 * r_ij
            denom = 1 - alpha_1 * beta_j * alpha_3 * r_ij
            self.numerator += num / denom
        self.denominator += 1


class TCMExaminationWrapper(ClickModelParamWrapper):
    # Shape: (max_rank,1)
    # Max rank in Yandex is 10

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """

        return [init_param_func(i, **kwargs) for i in xrange(MAX_DOCS_PER_QUERY)]

    def init_param_function(self, rank=0, **kwargs):
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
        return self.params[rank]

    def get_params_from_JSON(self, json_str):
        pass


class TCMRelevanceWrapper(ClickModelParamWrapper):
    # Shape: (i,j) pairs
    # N_docs that are on all result pages for query, so prob defaultdict of a defaultdict

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        init = partial(init_param_func, **kwargs)
        params = defaultdict(lambda: defaultdict(init))
        return params

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query and doc.
        """
        return self.params[session.query][session.web_results[rank]]

    def get_params_from_JSON(self, json_str):
        pass

    def __len__(self):
        return sum(len(i.keys()) for i in self.params.values())


class TCMIntentWrapper(ClickModelParamWrapper):
    # Shape: (1)

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query.
        """
        return self.params

    def get_params_from_JSON(self, json_str):
        pass


class TCMFreshnessWrapper(ClickModelParamWrapper):
    # Shape: (N_queries,N_docs)
    # See TCMRelevance for explanation

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given query and document.
        """

        return self.params

    def get_params_from_JSON(self, json_str):
        pass



