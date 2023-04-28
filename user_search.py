#! /usr/bin/env python3

#
#The user_search.py script is modified version of the user_search.py script found in Vespa's news example, available here: https://github.com/vespa-engine/sample-apps/blob/master/news/src/python/user_search.py 
#As noted in Vespa's repo, the file is offered under the Apache 2.0 license, a copy of which is available at the links below:
#- https://github.com/vespa-engine/sample-apps/blob/master/LICENSE
#- https://www.apache.org/licenses/LICENSE-2.0
#


import sys
import json
import urllib.parse
import requests

SCHEMA = "hotel_reviews"


def parse_embedding(hit_json):
    """Gets the values of the queried vector embedding from a Vespa JSON
    response
    Args:
        hit_json:   JSON response dict
    Returns:
        A list of floats (vector embedding)"""

    return hit_json["fields"]["embedding"]["values"]


def query_user_embedding(user_id):
    """Gets the values of the queried vector embedding from a Vespa JSON
    response
    Args:
        user_id:    document id used to query for a vector embedding
    Returns:
        A list of floats (vector embedding)"""

    yql = 'select * from sources {} where id contains "{}"'.format(
        SCHEMA, user_id)
    url = 'http://localhost:8080/search/?yql={}&hits=1'.format(
        urllib.parse.quote_plus(yql))
    result = requests.get(url).json()
    print(result["root"])
    return parse_embedding(result["root"]["children"][0])


def query_schema(user_vector, hits, filter):
    """Queries Vespa to perform an ANN search using a query vector
    Args:
        user_vector:    query vector used in the ANN search
        hits:   number of closest vectors to return
        filter: additional filters for the ANN query
    Returns:
        JSON response"""
        
    nn_annotations = [
        'targetHits:{}'.format(hits)
    ]
    nn_annotations = '{' + ','.join(nn_annotations) + '}'
    nn_search = '({}nearestNeighbor(embedding, review_embedding))'.format(
        nn_annotations)

    data = {
        'hits': hits,
        'yql': 'select id, text from sources {} where {} {}'.format(SCHEMA, nn_search, filter),
        'ranking.features.query(review_embedding)': str(user_vector),
        'ranking.profile': 'similarity'
    }
    return requests.post('http://localhost:8080/search/', json=data).json()


def main():
    user_id = sys.argv[1]
    hits = sys.argv[2] if len(sys.argv) > 2 else 10
    filter = sys.argv[3] if len(sys.argv) > 3 else ""

    user_vector = query_user_embedding(user_id)
    result = query_schema(user_vector, int(hits), filter)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
