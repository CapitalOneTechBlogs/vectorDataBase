#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

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
