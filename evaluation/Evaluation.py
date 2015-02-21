from abc import abstractmethod
from click_models.Constants import MAX_DOCS_PER_QUERY
from sklearn.metrics import mean_squared_error

class EvaluationMethod(object):

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, model, sessions, **kwargs):
        pass


class Loglikelihood(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        
        #NOTE(Luka) What to do with TCM with has different input datastructure

        loglikelihood = 0
        for session in sessions:
            log_click_probs = model.get_log_click_probs(session)
            loglikelihood += sum(log_click_probs) / len(log_click_probs)

        loglikelihood /= len(sessions)
        return loglikelihood 


class Perplexity(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        perplexity_at_rank = [0.0] * MAX_DOCS_PER_QUERY

        for session in sessions:
            log_click_probs = self.get_log_click_probs(session)
            for rank, log_click_prob in enumerate(log_click_probs):
                perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)

        perplexity_at_rank = [2 ** (-x / len(sessions)) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)



class ClickThroughRatePrediction(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        pass


class RelevancePrediction(EvaluationMethod):

    def __init__(self, true_relevances):
        # relevances should be a dict with structure relevances[query_id][region_id][url] -> relevance
        self.relevances = true_relevances
        super(self.__class__, self).__init__()

    def evaluate(self, model, sessions, **kwargs):
        pred_relevances = model.get_relevances(sessions)
        true_relevances = []
        for session in sessions:
            for rank, result in enumerate(session.web_results):
                print true_relevances
                if session.query in self.relevances:
                    if result in self.relevances[session.query]:
                        true_rel =  self.relevances[session.query][result]
                    else:
                        raise KeyError('URL not found in relevances document for this URL')
                else:
                    raise KeyError('Query not found in relevances document')
                true_relevances.append(true_rel)
        error = mean_squared_error(true_relevances, pred_relevances)
        return error


class RankingPerformance(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        pass

