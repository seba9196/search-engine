from datetime import datetime
import os
import requests
import logging
from bs4 import BeautifulSoup
from whoosh.fields import Schema, TEXT, ID, NUMERIC, DATETIME
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from argparse import ArgumentParser
from tqdm import tqdm
from whoosh.analysis import StandardAnalyzer

from whoosh import qparser
from whoosh import scoring
import gensim
from gensim import corpora
import numpy as np

logger = logging.Logger("default")


scp_series = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6.0",
    "6.5",
    "7.0",
    "7.5",
    "8.0",
    "8.5",
    "9.0",
    "9.5",
]

schema_dir = "indexdir"


class ScpItem:
    def __init__(self, name, url, rating, creator, creation_date, story, series):
        self.name = name
        self.url = url
        self.rating = rating
        self.creator = creator
        self.creation_date = creation_date
        self.story = story
        self.series = series

    # def __str__(self):
    #     return f"ScpItem({self.scp_name}, {self.url}, {self.rating}, {self.creator}, {self.creation_date}, {self.sotry}, {self.series})"


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    model = args.model
    score = args.scoring
    if args.train:
        train(model)

    query_string = args.query
    if query_string:
        results = search(model, score, query_string)
        for r in results:
            print(f"SCP: {r}, URL: https://scp-wiki.wikidot.com/{r}")


def search(model, score, query_string):
    if model == "doc2vec":
        if not os.path.exists("doc2vec.model"):
            print("Before searching you need to train the model")
            exit(1)
        model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")

        query_ter = query_string.split()
        inferred_vector = model.infer_vector(query_ter)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        return [s[0] for s in sims[:10]]

    if model == "vector":
        try:
            ix = open_dir(schema_dir)
        except:
            print("Before searching you need to generate the index")
            exit(1)

        if score == "TF_IDF":
            searcher = ix.searcher(weighting=scoring.TF_IDF())
        else:
            searcher = ix.searcher(weighting=scoring.BM25F(B=0.75, content_B=1.0, K1=1.5))

        with searcher as s:
            og = qparser.OrGroup.factory(0.85)
            query_parser = QueryParser("content", ix.schema, group=og)
            query = query_parser.parse(query_string)
            corrected = s.correct_query(query, query_string)

            if corrected.query != query:
                response = input(f'Did you mean: "{corrected.string}"? (y/n): ')
                if response == "y":
                    query = corrected.query
            results = s.search(query, limit=10, terms=True)
            if len(results) == 0:
                print("no results found!")
            else:
                return [r["scp_name"] for r in results[:10]]


def train(model):
    if model == "doc2vec":
        train_doc2vec()
    if model == "vector":
        generate_index()


def train_doc2vec():
    corpus = get_corpus_documents()
    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=50, min_count=2, epochs=100, seed=1
    )
    model.build_vocab(corpus)
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )
    model.save("doc2vec.model")
    print("Model training completed")


def generate_index():
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    schema = Schema(
        scp_name=TEXT(stored=True),
        url=ID(stored=True),
        rating=NUMERIC(stored=True),
        creator=TEXT(stored=True),
        creation_date=DATETIME(stored=True),
        content=TEXT(spelling=True, analyzer=StandardAnalyzer()),
        series=NUMERIC(stored=True),
    )

    ix = create_in(schema_dir, schema)

    writer = ix.writer()

    items = get_scp_items()
    with tqdm(total=8064, desc="Indexing items") as pbar:
        for item in items:
            writer.add_document(
                scp_name=item.name,
                url=item.url,
                rating=item.rating,
                creator=item.creator,
                creation_date=item.creation_date,
                content=item.story,
                series=item.series,
            )
            pbar.update(1)

    writer.commit()
    return ix


def get_corpus_documents(series=scp_series):
    documents = []
    for item in get_scp_items(series):
        print("processing item", item.name)
        tokens = gensim.utils.simple_preprocess(item.story)
        documents.append(gensim.models.doc2vec.TaggedDocument(tokens, [item.name]))
    return documents


def get_scp_items(series=scp_series):
    for s in series:
        url = f"https://scp-data.tedivm.com/data/scp/items/content_series-{s}.json"
        response = requests.get(url)
        scp_metadata = response.json()

        for item_id, item_data in scp_metadata.items():
            creator = item_data["creator"]
            url = item_data["url"]
            html = item_data["raw_content"]
            text = clean_html(html)
            rating = item_data["rating"]
            created = item_data["created_at"]

            try:
                rating = int(rating)
            except:
                print(f"found invalid int for rating {rating}")

            try:
                series_number = int(float(s))
            except:
                print(f"invalid nunber for series {s}")

            try:
                creation_date = datetime.strptime(created, r"%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                print(f"Invalid date format: {created}, error: {e}")
                print(f"invalide date format: {created}")

            yield ScpItem(
                name=item_id,
                url=url,
                rating=rating,
                creator=creator,
                creation_date=creation_date,
                story=text,
                series=series_number,
            )


# Funzione per pulire l'HTML
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    # Rimuovi i tag di script e stile
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    # Ottieni il testo pulito
    clean_text = soup.get_text(separator=" ")
    # Rimuovi spazi multipli
    clean_text = " ".join(clean_text.split())
    return clean_text


def setup_argparse():
    parser = ArgumentParser(description="Script to search SCPs")

    parser.add_argument(
        "model",
        help="choose the model to use",
        choices=["vector", "doc2vec"],
    )

    parser.add_argument(
        "scoring",
        help="if you use the vector model you choose the scoring algorithm to use",
        choices=["BM25F", "TF_IDF"],
    )

    parser.add_argument(
        "query",
        help="insert the query you want to search",
        default=None,
        nargs="?",
    )

    parser.add_argument("--train", action="store_true", help="train the model")
    return parser


if __name__ == "__main__":
    main()
