import sys
from session.Session import Session, Result
import numpy as np
from collections import defaultdict


def transform_to_tasks(sessions):
    # time_interval = 430  NOTE(Luka): Taken from Piwowarski 2009, Mining User Web Search Activity... according to TCM Paper
    # Should be in seconds however Yandex click log does not add this information..
    # This means all queries with same session_id will be a task.
    # Also have no way to measure similarity between queries.
    # However we can assume that Yandex has made session_ids already into tasks according to website
    return sessions.values()


def parse_yandex_sessions(sessions_filename, max_sessions=None):
    """
        Parse Yandex-sessions formatted as found here: 'http://imat-relpred.yandex.ru/en/datasets'

        Two possible formats:

        Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs

        Click action
        SessionID TimePassed TypeOfAction URLID

        Returns list of sessions

    """
    sessions_file = open(sessions_filename, "r")
    session_dict = defaultdict(list)
    count = 0
    another_counter = 0
    for line_n, line in enumerate(sessions_file):

        # NOTE(Luka): Limit the amount of lines to ease the development work because the whole file has 14.5 Million Sessions-lines and 19.5 Million Clicks-lines.
        session_str = line.strip().split('\t')
        session_id = session_str[0]
        time = session_str[1]
        type_action = session_str[2]
        if type_action is "Q":
            query = session_str[3]
            region = session_str[4]
            docs = session_str[5:]

            session = Session(session_id, query, region = region)
            for doc in docs:
                web_result = Result(doc, -1, 0)
                session.add_web_result(web_result)

            if count >= max_sessions:
                break
            
            
            if session_dict[session_id]:
                last_session_same_id = session_dict[session_id][-1]
                if last_session_same_id.query  == session.query:
                    if [web.object for web in last_session_same_id.web_results] == [web.object for web in session.web_results]:
                       continue

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
    
    return [s for t in session_dict.values() for s in t]

def parse_yandex_relevances(relevances_filename):
    relevances = dict()
    with open(relevances_filename, "r") as f:
        for row in f:
            query_id, region_id, url, rel = row.split('\t')
            if query_id not in relevances:
                relevances[query_id] = dict()
            query = relevances[query_id] 
            
            if region_id not in query:
                query[region_id] = dict()
            region = query[region_id]
            
            region[url] = rel



    return relevances


