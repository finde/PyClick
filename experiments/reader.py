import sys
from click_models.BBM import BBM
from session.Session import Session, Result
import numpy as np

def train_test_split(sessions, split_fraction):
    sessions = np.array(sessions)
    mask = np.random.rand(*sessions.shape)
    train = mask < split_fraction
    train_sessions = sessions[train]
    test_sessions = sessions[np.logical_not(train)]
    return train_sessions, test_sessions

def parse_yandex_sessions(sessions_filename, max_sessions = None, split_fraction = 0.75):
    """
        Parse Yandex-sessions formatted as found here: 'http://imat-relpred.yandex.ru/en/datasets'

        Two possible formats:

        Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs

        Click action
        SessionID TimePassed TypeOfAction URLID

        Returns list of sessions

    """
    sessions_file = open(sessions_filename,"r")
    sessions = []
    all_clicks = []
    for line_n,line in enumerate(sessions_file):

        #NOTE(Luka): Limit the amount of lines to ease the development work because the whole file has 14.5 Million Sessions-lines and 19.5 Million Clicks-lines.
        session_str = line.strip().split('\t')
        session_id = session_str[0]
        time = session_str[1]
        type_action = session_str[2]
        if type_action is "Q":
            query = session_str[3]
            region = session_str[4]
            docs = session_str[5:]

            session = Session(query)
            for doc in docs:
                web_result = Result(doc, -1, 0)
                session.add_web_result(web_result)
            
            if sessions and not [click for click in sessions[-1].get_clicks() if click]:
                sessions.pop()
            if len(sessions) >= max_sessions:
                break
            sessions.append(session)

        if type_action is "C":
            doc = session_str[3]
            #NOTE(Luka): Only look at last session, don't look at session_id
            for result in sessions[-1].web_results:
                if result.object == doc:
                    result.click = 1
                    all_clicks.append((sessions[-1],result))
                    #print "Click found!"
                    break
            else:
                #NOTE(Luka): Click not found might be artifact or because clicked result is after clipped results.
                #print "Could not find {} for session {}".format(doc,session_id)
                pass
    
    train_set, test_set = train_test_split(sessions,split_fraction)
    return train_set, test_set

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print "The format that should be used for executing the reader is `python YandexReader.py <query_log>`"
        sys.exit()

    max_sessions = 1000
    fn = sys.argv[1]
    train_set, test_set = parse_yandex_sessions(fn,max_sessions)
    print "Train set:",train_set.size,"sessions loaded."
    print "Test set:",test_set.size,"sessions loaded."

