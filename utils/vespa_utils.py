import pandas as pd
import json


def data_to_vespa_format(
    df,
    embeddings,
    path="./data/reviews.json",
    schema="hotel_reviews"
):
    """Generates a Vespa-feedable JSON file using record information from a df and
    an `np.ndarray` of vector embeddings
    Args:
        df: Dataframe containing metadata for each record
        embeddings: `np.ndarray` containing vector embeddings for each record
        path:   filepath to save outputted JSON file
        schema: name of the schema used in the Vespa database"""
        
    num_docs = 0
    with open(path, 'w') as f:
        f.write('[\n')
        for index, row in df.iterrows():

            if num_docs:
                f.write(",\n")

            fields = {
                "id": index,
                "text": row['text'],
                "label": row['label'],
                "embedding": {
                    "values": embeddings[index].tolist()
                }
            }

            content = json.dumps({
                "put": f"id:{schema}:{schema}::{int(index)}",
                "fields": fields
            })

            f.write(content)
            num_docs += 1

        f.write(']\n')

    print(f"{num_docs} successfully saved at {path}")
