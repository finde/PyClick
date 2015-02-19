
class EvaluationMethod:

    @abstractmethod
    def evaluate(model, sessions, **kwargs):
        pass


class Loglikelihood(EvaluationMethod):

    @staticmethod
    def evaluate(model, sessions, **kwargs):
        
        #NOTE(Luka) What to do with TCM with has different input datastructure

        loglikelihood = 0
        for session in sessions:
            log_click_probs = model.get_log_click_probs(session)
            loglikelihood += sum(log_click_probs) / len(log_click_probs)

        loglikelihood /= len(sessions)
        return loglikelihood 


class Perplexity(EvaluationMethod):

    @staticmethod
    def evaluate(model, sessions, **kwargs):
        perplexity_at_rank = [0.0] * MAX_DOCS_PER_QUERY

        for session in sessions:
            log_click_probs = self.get_log_click_probs(session)
            for rank, log_click_prob in enumerate(log_click_probs):
                perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)

        perplexity_at_rank = [2 ** (-x / len(sessions)) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)



class ClickThroughRatePrediction(EvaluationMethod):

    @staticmethod
    def evaluate(model, sessions, **kwargs):
        pass


class RelevancePrediction(EvaluationMethod):

    @staticmethod
    def evaluate(model, sessions, **kwargs):
        pass

class RankingPerformance(EvaluationMethod):

    @staticmethod
    def evaluate(model, sessions, **kwargs):
        pass

