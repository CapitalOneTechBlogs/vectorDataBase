## This repository is for primarily documentation and used for referencing code in Tech blogs, Medium articles and other publications. The repo does not have any project admin and will not be supported or go through maintenance.

_Code uses dataset [Trip Advisor Hotel Reviews @ Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)_

##
# Sample Vespa Pipeline

A simple pipeline for taking vectors from a Character CNN (CharCNN) model, feeding them to a Vespa vector database, and querying the database using ANN.

## Installations:

### Conda Environment

To setup your Conda environment to run the `get_vectors.ipynb` notebook, please run the following:

`conda create --name vespa`

Now, activate your environment with the following command:

`conda activate vespa`

Now, add in the libraries you will need:
`conda install requests pandas tensorflow numpy ipykernel`

Export the environment to a yaml file for subsequent use:
`conda env export > environment.yml`

Make the conda environment visible to your notebook server:
ipython kernel install --user --name=vespa

### Docker Installation

Docker is needed to run a local Vespa application. Please install Docker Desktop [here](https://docs.docker.com/desktop/#download-and-install)

### Vespa Setup

Once Docker is installed, the Vespa CLI will be needed. You can install it using the brew command `brew install vespa-cli` or through the download links [here](https://github.com/vespa-engine/vespa/releases)

Then, we need to start a Vespa container. First, pull the vespa image from Docker hub:

`docker pull vespaengine/vespa`

Then, start the actual Vespa container:

```
docker run -m 10G --detach --name vespa --hostname vespa-tutorial \
  --publish 8080:8080 --publish 19071:19071 --publish 19092:19092 \
  vespaengine/vespa
```

Finally, deploy your application using the following:

`vespa deploy --wait 300 app`


## How to Use:

### Feeding Vectors

Feeding vectors to Vespa is best done through the `vespa=feed-client`. You can download the latest client [here](https://search.maven.org/artifact/com.yahoo.vespa/vespa-feed-client-cli)

Once unzipped, the client can be used to feed JSON files to Vespa. A JSON file of 10 documents is already included in `data/reviews.json`, but the actual process for generating these documents can be seen in the `get_vectors.ipynb` Jupyter notebook. The vectors themselves are generated from the Keras model saved in the `model` directory.

To batch feed the `data/reviews.json` documents to Vespa, run the following command:

```
./vespa-feed-client-cli/vespa-feed-client \
  --verbose --file ./data/reviews.json --endpoint http://localhost:8080
```

### Querying the Database

You can query your Vespa application from the command line. For example, running the command...

`vespa query -v 'yql=select * from hotel_reviews where true'`

...will return all documents in the schema.

Querying using approximate nearest neighbor search (ANN) over the command line is much more complex. To simplify the process, the `user_search.py` script is included to easily make an ANN query using an existing document's embedding vector as your query vector. For example, if you run the following:

`python user_search.py 2 3`

Then an ANN search will be performed to find the 3 nearest vectors to the embedding vector of document ID=2.

---

_Note: The `user_search.py` script is modified version of the `user_search.py` script found in Vespa's `news` example, found [here](https://github.com/vespa-engine/sample-apps/blob/master/news/src/python/user_search.py)_
