#!/bin/bash

num_query=1;
cat query.txt | while IFS= read -r line; do
echo "QUERY $num_query: $line";
echo "vector-BM25F:";
yes 1 | python3 search_engine.py vector-BM25F "$line";
echo "vector-TFIDF:";
yes 1 | python3 search_engine.py vector-TFIDF "$line";
echo "doc2vec:";
python3 search_engine.py doc2vec "$line";
echo "============================"
((num_query=num_query+1));
done | tee results.txt
