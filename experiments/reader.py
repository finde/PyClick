import sys
from session.Session import Session, Result
import numpy as np
from collections import defaultdict

def transform_to_tasks(sessions):
    #time_interval = 430  NOTE(Luka): Taken from Piwowarski 2009, Mining User Web Search Activity... according to TCM Paper
                        # Should be in seconds however Yandex click log does not add this information..
                        # This means all queries with same session_id will be a task.
                        # Also have no way to measure similarity between queries..
    return sessions.items()


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
    session_dict = defaultdict(list)
    count = 0
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
            
            if count >= max_sessions:
                break
            
            session_dict[session_id].append(session)
            count += 1

        if type_action is "C":
            doc = session_str[3]
            #NOTE(Luka): Only look at last session, don't look at session_id
            for result in session_dict[session_id][-1].web_results:
                if result.object == doc:
                    result.click = 1

                    #print "Click found!"
                    break
            else:
                #NOTE(Luka): Click not found might be artifact or because clicked result is after clipped results.
                #print "Could not find {} for session {}".format(doc,session_id)
                pass
    
    return session_dict

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print "The format that should be used for executing the reader is `python YandexReader.py <query_log>`"
        sys.exit()

    max_sessions = 1000
    fn = sys.argv[1]
    sessions_dict = parse_yandex_sessions(fn,max_sessions)
    tasks = transform_to_tasks(sessions_dict)
    print sessions_dict['67']
    print '#Tasks:\t',len(tasks)
