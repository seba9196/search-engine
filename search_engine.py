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

import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim import similarities
import numpy as np

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
            # preprocessed_query = " ".join(preprocess(query))
            # print(preprocessed_query)
            # query = QueryParser("content", ix.schema).parse(preprocessed_query)
            query = QueryParser("content", ix.schema).parse(query)
            results = searcher.search(query, limit=10, terms=True)
            if len(results) == 0:
                print("no results found!")
            else:
                for hit in results:
                    print(hit["scp_name"], hit["url"])
                    print(hit.matched_terms())

    if args.word2vec:
        train_corpus = list(read_corpus())
        dictionary_t = create_dictionary(train_corpus)
        bow_corpus = [
            dictionary_t.doc2bow(text) for text in [t.words for t in train_corpus]
        ]
        tfidf = gensim.models.TfidfModel(bow_corpus)
        index_t = similarities.SparseMatrixSimilarity(
            tfidf[bow_corpus], num_features=len(dictionary_t)
        )
        # index_t.index.shape
        query = input("Inserisci query per word2vec: ")
        query = query.split()
        for d in get_closest_n(query, 10, dictionary_t, index_t, tfidf, train_corpus):
            print(f"{d[1]:.3f}: {d[0].tags[0]}")

    if args.doc2vec_train:
        corpus = get_corpus_documents(["1"])
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

    if args.doc2vec:
        if not os.path.exists("doc2vec.model"):
            print("Before searching you need to train the model")
            exit(1)
        model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")

        query_ter = input("Inserisci query per doc2vec: ")
        query_ter = query_ter.split()
        inferred_vector = model.infer_vector(query_ter)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        for index in range(10):
            most_similar_key, similarity = sims[index]
            print(f"{most_similar_key}: {similarity:.4f}")


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

    items = get_scp_items()
    with tqdm(total=8064, desc="Indexing items") as pbar:
        for item in items:
            writer.add_document(
                scp_name=item.name,
                url=item.url,
                rating=item.rating,
                creator=item.creator,
                creation_date=item.creation_date,
                content=preprocess(item.story),
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


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = remove_stopwords(tokens)  # Rimozione stopwords
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
    parser.add_argument(
        "--word2vec",
        action="store_true",
        help="use word2vec model, usare il comando senza la query",
    )
    parser.add_argument(
        "--doc2vec-train",
        action="store_true",
        help="use doc2vec model, usare il comando senza la query",
    )

    parser.add_argument(
        "--doc2vec",
        action="store_true",
        help="use doc2vec model, usare il comando senza la query",
    )
    return parser


def read_corpus(tokens_only=False):
    scp_series = ["1"]
    doc_id = 0
    for s in scp_series:
        print(f"processing series {s}")
        url = f"https://scp-data.tedivm.com/data/scp/items/content_series-{s}.json"
        response = requests.get(url)
        scp_metadata = response.json()

        for item_id, item_data in scp_metadata.items():
            scp_name = item_id
            url = item_data["url"]
            html = item_data["raw_content"]
            text = clean_html(html)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                doc_id += 1
                yield
        print(f"end of processing series {s}")


def create_dictionary(t_docs):
    "create dictionary of words in preprocessed corpus"
    docs = [t.words for t in t_docs]
    dictionary = corpora.Dictionary(docs)
    return dictionary


def get_closest_n(query_document, n, dictionary, index, tfidf, t_corpus):
    """get the top matching docs as per cosine similarity
    between tfidf vector of query and all docs"""
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]

    top_idx = sims.argsort()[-1 * n :][::-1]
    top_val = np.sort(sims)[-1 * n :][::-1]
    # return most similar documents and the related similarities
    return [(t_corpus[i[0]], i[1]) for i in zip(top_idx, top_val)]


if __name__ == "__main__":
    main()
