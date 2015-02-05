#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper, ClickModel, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY


__author__ = 'Ilya Markov'


class TCM(ClickModel):

    def init_params(self, init_param_values):
        self.param_helper = TCMParamHelper()
        params = {
                #PARAMETERS
                }

        return params

    def get_p_click(self, param_values):
        pass

    def predict_click_probs(self, session):
        pass

    def from_JSON(self, json_str):
        param_helper_backup = self.param_helper
        super(UBM, self).from_JSON(json_str)
        self.param_helper = param_helper_backup

    @staticmethod
    def get_prior_values():
        return {UBMRelevance.NAME: 0.5,
                UBMExamination.NAME: 0.5}


class TCMParamHelper(object):
    pass

class TCMParam(ClickModelParam):
    """
        Parameter of a general click model
    """
    def __init__(self, init_value, **kwargs):
        super(TCMParam, self).__init__(init_value)
        if 'param_helper' in kwargs:
            self.param_helper = kwargs['param_helper']
        else:
            self.param_helper = TCMParamHelper()


class TCMRelevance(UBMParam):
    """
        Probability of relevance/attractiveness: rel = P(A = 1 | E = 1).
    """
    NAME = "rel"

    def update_value(self, param_values, click):
        pass

class TCMExamination(ClickModelParam):
    """
        Examination probability: exam = P(E = 1).
    """
    NAME = "exam"

    def update_value(self, param_values, click):
        pass

class TCMExaminationWrapper(ClickModelParamWrapper):
    pass

class TCMRelevanceWrapper(RelevanceWrapper):
    pass

class UBMRelevanceWrapperRel(RelevanceWrapperRel):
    pass
