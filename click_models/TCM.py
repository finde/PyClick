#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper, ClickModel, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY

MAX_ITERATIONS = 40


__author__ = 'Ilya Markov'

class TCM(ClickModel):

    def init_params(self, init_param_values):
        self.param_helper = TCMParamHelper()
        params = {
                TCMExamination.NAME: TCMExaminationWrapper(init_param_values, TCMExamination.default(), param_helper = self.param_helper),
                TCMRelevance.NAME: TCMRelevanceWrapper(init_param_values, TCMRelevance.default(),param_helper = self.param_helper),
                TCMIntent.NAME: TCMIntentWrapper(init_param_values, TCMIntent.default(),param_helper = self.param_helper),
                TCMAnother.NAME: TCMAnotherWrapper(init_param_values, TCMAnother.default(),param_helper = self.param_helper),
                TCMFreshness.NAME: TCMFreshnessWrapper(init_param_values, TCMFreshness.default(),param_helper = self.param_helper)
                }
        return params

    def train(self, tasks):
        
        if len(tasks) <= 0:
            print >>sys.stderr, "The number of training sessions is zero."
            return

        
        diff = float('inf')
        treshold = 0.0001
        while diff > treshold:

            #E-Step
            #In the E-Step, we compute the marginal posterior distribution of each hidden variable to associate parameters that we introduced. The computation is performed based on the parameter values updated in the previous iteration, which are further discussed in Section 5.1. 

            #M-Step
            #In the M-Step, all posterior probabilities associated with the same param- eter are averaged to update the parameters. 
            self.params = self.get_updated_params(tasks, self.params)



    def get_updated_params(self, tasks, priors):
        return self.params

    def get_p_click(self, param_values):
        pass

    def predict_click_probs(self, session): #NOTE(Luka): Might become task instead of session
        pass

    def from_JSON(self, json_str):
        param_helper_backup = self.param_helper
        super(UBM, self).from_JSON(json_str)
        self.param_helper = param_helper_backup

    @staticmethod
    def get_prior_values():
        return {
                TCMRelevance.NAME: 0.5,
                TCMExamination.NAME: 0.5,
                TCMIntent.NAME: 0.5,
                TCMAnother.NAME: 0.5,
                TCMFreshness.NAME: 0.5,
                }



class TCMParamHelper(object):
    pass

class TCMRelevance(ClickModelParam):
    """
        Probability of relevance: rel = P(R_ij = 1) = r_d.
    """
    NAME = "rel"

    def update_value(self, param_values):
        pass

class TCMExamination(ClickModelParam):
    """
        Examination probability: exam = P(E_ij = 1). beta_j
    """
    NAME = "exam"

    def update_value(self, param_values):
        pass

class TCMIntent(ClickModelParam):
    """
        Matches user intent probability: intent = P(M_i = 1). alpha1
    """
    NAME = "intent"

    def update_values(self, param_values):
        pass

class TCMAnother(ClickModelParam):
    """
        Whether user submits other query after i: another = P(N_i = 1 | M_i = 1). alpha2
    """
    NAME = "another"

    def update_values(self, param_values):
        pass

class TCMFreshness(ClickModelParam):
    """
        Freshness of document: fresh = P(F_ij | H_ij). alpha3
        H_ij = Previous examination of doc at rank j in session (Binary)
    """
    NAME = "fresh"

    def update_values(self, param_values):
        pass


class TCMExaminationWrapper(ClickModelParamWrapper):

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    def get_params_from_JSON(self, json_str):
        pass


class TCMRelevanceWrapper(ClickModelParamWrapper):

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)

    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    def get_params_from_JSON(self, json_str):
        pass

class TCMIntentWrapper(ClickModelParamWrapper):

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    def get_params_from_JSON(self, json_str):
        pass


class TCMAnotherWrapper(ClickModelParamWrapper):

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    def get_params_from_JSON(self, json_str):
        pass


class TCMFreshnessWrapper(ClickModelParamWrapper):

    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        return init_param_func(**kwargs)


    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    def get_params_from_JSON(self, json_str):
        pass



