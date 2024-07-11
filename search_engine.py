from datetime import datetime
import os
import requests
import logging
from bs4 import BeautifulSoup
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID, NUMERIC, DATETIME
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from argparse import ArgumentParser

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
            query = QueryParser("content", ix.schema).parse(query)
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
        content=TEXT(analyzer=StemmingAnalyzer()),
        series=NUMERIC(stored=True),
    )
    ix = create_in(schema_dir, schema)
    writer = ix.writer()
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
                content=text,
                series=series_number,
            )
    writer.commit()
    return ix


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


if __name__ == "__main__":
    main()
