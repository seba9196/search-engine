from whoosh.index import create_in
from whoosh.fields import *
import os

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
if not os.path.exists("indexdir"):
   os.mkdir("indexdir")
ix = create_in("indexdir", schema)
writer = ix.writer()

from nltk.corpus import brown
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize
import string

for doc in inaugural.fileids():
    raw = inaugural.raw(doc)
    #writer.add_document(title=doc, path="/a", content=raw)
    writer.add_document(title=doc, content=raw)

writer.commit()

from whoosh.qparser import QueryParser
with ix.searcher() as searcher:
    while True:
        i = input("Cosa vuoi cercare?(premi q per uscire) ")
        if i == "q":
            break
        else:
            query = QueryParser("content", ix.schema).parse(i)
            results = searcher.search(query)
            if len(results) == 0:
                print("NESSUN RISULTATO TROVATO")
            else:
                for i in results:
                    print(i)
