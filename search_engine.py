from datetime import datetime
import os
import requests
import logging
from bs4 import BeautifulSoup
from whoosh.fields import Schema, TEXT, ID, NUMERIC, DATETIME
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from argparse import ArgumentParser

import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.Logger("default")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    if args.gen_index:
        ix = generate_index()
    else:
        try:
            ix = open_dir(schema_dir)
        except:
            print("Before searching you need to generate the index")
            exit(1)

    if args.query:
        with ix.searcher() as searcher:
            query = args.query
            preprocessed_query = " ".join(preprocess(query))
            print(preprocessed_query)
            query = QueryParser("content", ix.schema).parse(preprocessed_query)
            results = searcher.search(query, limit=10, terms=True)
            if len(results) == 0:
                print("no results found!")
            else:
                for hit in results:
                    print(hit["scp_name"], hit["url"])
                    print(hit.matched_terms())


def generate_index():
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    schema = Schema(
        scp_name=TEXT(stored=True),
        url=ID(stored=True),
        rating=NUMERIC(stored=True),
        creator=TEXT(stored=True),
        creation_date=DATETIME(stored=True),
        content=TEXT(),
        series=NUMERIC(stored=True),
    )
    ix = create_in(schema_dir, schema)
    writer = ix.writer()

    indexed_items = 0
    for s in scp_series:
        print(f"processing series {s}")
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

            writer.add_document(
                scp_name=item_id,
                url=url,
                rating=rating,
                creator=creator,
                creation_date=creation_date,
                content=preprocess(text),
                series=series_number,
            )
        print(f"indexed {len(scp_metadata)} items for series {s}")
        indexed_items += len(scp_metadata)
    writer.commit()
    print(f"indexed items: {indexed_items}")
    return ix


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = [
        word for word in tokens if word not in string.punctuation and not word.isdigit()
    ]  # Rimozione punteggiatura e numeri
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatizzazione
    return tokens


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]


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
        "query",
        help="insert the query you want to search",
        default=None,
        nargs="?",
    )
    parser.add_argument("--gen-index", action="store_true", help="index the content")
    return parser


if __name__ == "__main__":
    main()
