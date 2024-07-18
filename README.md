# Search-Engine

Progetto per il corso universitario di Gestione dell'Informazione 2023/2024

# Installazione ed Esecuzione

Per installare l'applicazione eseguire i seguenti comandi
    
    python3 -m venv env
    
    source env/bin/activate
    
    pip install -r requirements.txt

Per eseguire l'applicazione utilizzare il seguente comando
    
    python3 search_engine.py [-h] [--train] {vector-BM25F,vector-TFIDF,doc2vec} [query]

Per eseguire test sull'applicazione
    
    bash test.py

# Utilizzo
Sono presenti 3 comandi principali per eseguire l'applicazione:
* vector-BM25F --> ricerca utilizzando l'inverted index e BM25F come algoritmo di ranking
* vector-TFIDF --> ricerca utilizzando l'inverted index e TF_IDF come algoritmo di ranking
* doc2vec --> ricerca e ranking utilizzando il modello doc2vec

Infine il comando **--train** permette di allenare il modello per la ricerca doc3vec e il comando **-h** per richiamare l'help.
  
Utilizzando il modello vettoriale l'applicazione accetta query di tipo:
* multiple word query
* phrase query
* boolean query
* ranges query

# QUERY
Elenco di query per mostrare il **default query language** di Whoosh:
* Gli articoli che parlano di **ghost** -> "ghost"
* Gli articoli che parlano di **nigh*** -> "nigh*"
* Gli articoli che parlano di **ca?tle** -> "ca?tle"
* Gli articoli che contengono le parole **spacewalking** e **countless** a distanza di **5 parole** -> "spacewalking countless"~5
* Gli articoli che contengono la frase " " -> " "
* Gli articoli scritto dall'autore **smapti** -> "creator:smapti"
* Gli articolu che contengono **ghost** o **castle** ma non **night** -> "ghost OR castle NOT night"
* Gli articoli che parlano di **night** nella **serie 1** -> "night AND series:1"
* Gli articoli scitti nelle **date tra 2010-11-04 e 2024-06-01** -> "creation_date:[20101104 TO 20240601]"
* Gli articoli che parlano di **vampire** scritti nelle **date tra 2011-11-14 e 2024-01-15** e con **rating compreso tra 100 e 500** -> "vampire AND creation_date:[20111114 TO 20240115] AND rating:[100 TO 500]"

Elenco di query utilizzate per eseguire i benchmark:

    Esplora SCP con caratteristiche bioingegneristiche avanzate
    Query: "Explore SCPs with advanced bioengineering features"

    Racconti di recupero e contenimento di SCP pericolosi
    Query: "Accounts of retrieval and containment of dangerous SCPs"

    Rapporti di ricerca su SCP con anomalie temporali
    Query: "Research reports on SCPs with temporal anomalies"

    Esperimenti documentati su SCP con proprietà chimiche uniche
    Query: "Documented experiments on SCPs with unique chemical properties"

    Analisi delle interazioni tra SCP e ambienti urbani
    Query: "Analysis of interactions between SCPs and urban environments"

    Studi sui SCP che influenzano la percezione umana
    Query: "Studies on SCPs that affect human perception"

    Descrizioni di SCP con capacità di alterare la realtà
    Query: "Descriptions of SCPs with reality-altering abilities"

    Resoconti storici di incidenti causati da SCP contenuti in modo inadeguato
    Query: "Historical accounts of incidents caused by inadequately contained SCPs"

    Esplorazioni di siti anomali legati a SCP
    Query: "Explorations of anomalous sites related to SCPs"

    Rapporti su SCP che esibiscono comportamenti predatori
    Query: "Reports on SCPs exhibiting predatory behaviors"

# Benchmark
