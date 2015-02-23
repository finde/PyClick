from abc import abstractmethod
from click_models.Constants import MAX_DOCS_PER_QUERY
from sklearn.metrics import mean_squared_error
import math
from click_models.TCM import TCM
import collections

class EvaluationMethod(object):

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, model, sessions, **kwargs):
        pass


class Loglikelihood(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        
        if model.__class__ == TCM:
            len_sessions = 0
            loglikelihood = 0
            tasks = model._transform_sessions(sessions)
            for task in tasks:
                freshness = model.get_previous_examination_chance(task)
                for s_idx, session in enumerate(task):
                    log_click_probs = model.get_log_click_probs(session, freshness = freshness[s_idx])
                    loglikelihood += sum(log_click_probs) / len(log_click_probs)

            loglikelihood /= len(sessions)
        
        else:
            loglikelihood = 0
            for session in sessions:
                log_click_probs = model.get_log_click_probs(session)
                loglikelihood += sum(log_click_probs) / len(log_click_probs)

            loglikelihood /= len(sessions)
        return loglikelihood 


class Perplexity(EvaluationMethod):

    def evaluate(self, model, sessions, **kwargs):
        perplexity_at_rank = [0.0] * MAX_DOCS_PER_QUERY

        if model.__class__ == TCM:
            len_sessions = 0
            loglikelihood = 0
            tasks = model._transform_sessions(sessions)
            for task in tasks:
                freshness = model.get_previous_examination_chance(task)
                for s_idx, session in enumerate(task):
                    log_click_probs = model.get_log_click_probs(session, freshness = freshness[s_idx])
                    for rank, log_click_prob in enumerate(log_click_probs):
                        perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)

                    
            perplexity_at_rank = [2 ** (-x / len(sessions)) for x in perplexity_at_rank]
            perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        
        else:
            for session in sessions:
                log_click_probs = model.get_log_click_probs(session)
                for rank, log_click_prob in enumerate(log_click_probs):
                    perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)

            perplexity_at_rank = [2 ** (-x / len(sessions)) for x in perplexity_at_rank]
            perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return perplexity, perplexity_at_rank



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
                if session.query in self.relevances:
                    if session.region in self.relevances[session.query]:
                        if result.object in self.relevances[session.query][session.region]:
                            true_rel =  self.relevances[session.query][session.region][result.object]
                        else:
                            true_rel = 0
                    else:
                        true_rel = 0        
                else:
                    true_rel = 0    
                true_relevances.append(true_rel)
        error = mean_squared_error(true_relevances, pred_relevances)
        return error


class RankingPerformance(EvaluationMethod):

    def __init__(self, true_relevances):
        # relevances should be a dict with structure relevances[query_id][region_id][url] -> relevanc
        self.relevances = true_relevances
        super(self.__class__, self).__init__()


    def evaluate(self, model, sessions, **kwargs):
        counter = collections.Counter([session.query for session in sessions])
        useful_sessions = [query_id for query_id in counter if counter[query_id] >= 5 and query_id in self.relevances]
        
        total_ndcg = 0

        sessions_dict = dict()
        for session in sessions:
            if session.query in useful_sessions:
                if not session.query in sessions_dict:
                    sessions_dict[session.query] = []
                    
                sessions_dict[session.query].append(session)
        
        predicted = dict()
        for query_id in useful_sessions:
            sessions = sessions_dict[query_id]
            pred_rels = model.get_relevances(sessions)

            i = 0
            for session in sessions:
                for result in session.web_results:
                    predicted[result.object] = pred_rels[i]
                    i += 1

            ranking = sorted(predicted.values(),reverse = True)


            #REGION HACK. Now does first region how to remove? Use query/region pairs for indexing of the queries?
            rel = self.relevances[query_id]

            ideal_ranking = sorted(rel[rel.keys()[0]].values(),reverse = True)
            dcg = self.dcg(ranking)
            idcg = self.dcg(ideal_ranking)
            ndcg = dcg / idcg

            total_ndcg += ndcg
        return total_ndcg / len(useful_sessions)
            

    def dcg(self, ranking):
        dcg = ranking[0]
        for i,doc in enumerate(ranking[1:5]):
            dcg += ranking[i]/math.log(i+2,2)
        return dcg

                
        # for all queries occuring more than 10 times and have rankings in true_relevances 
        #   Retrieve all sessions for given query
        #   Get relevance for urls
        #   compute NDCG@5
        #   average NDCG@5


