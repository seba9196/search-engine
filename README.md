# SCP search engine

Progetto per il corso universitario di Gestione dell'Informazione 2023/2024 sviluppato da Roberto Bertelli e Sebastiano Benatti 

Il progetto è un search enginge per gli item della [SCP Foundation](https://scp-wiki.wikidot.com/).


## SCP foundation

![SCP logo](SCP_logo.png)


La Fondazione SCP è un'organizzazione fittizia presente in progetti di scrittura collaborativa, il cui scopo è garantire la sicurezza, il contenimento e la protezione di anomalie (SCP). Queste anomalie possono essere oggetti, entità o fenomeni che sfidano le leggi naturali e rappresentano una minaccia per l'umanità. La missione della Fondazione è studiare e comprendere queste anomalie per prevenire danni.
Oggetti SCP

Gli oggetti SCP sono le singole anomalie contenute e studiate dalla Fondazione SCP. Ogni SCP viene assegnato un numero unico e documentato con una descrizione dettagliata, procedure di contenimento e informazioni rilevanti sulle sue caratteristiche e comportamenti.


## Installazione ed Esecuzione del search engine

Per installare l'applicazione eseguire i seguenti comandi
    
    python3 -m venv env
    
    source env/bin/activate
    
    pip install -r requirements.txt

Per eseguire l'applicazione utilizzare il seguente comando
    
    python3 search_engine.py [-h] [--train] {BM25F,TFIDF,doc2vec} [query]

Per eseguire i benchmark sull'applicazione
    
    bash test.py

verrà generato un file chiamato `results.txt` contenente i risultati delle query di benchmark

## Utilizzo

Sono presenti 3 parametri principali per eseguire l'applicazione:
* `BM25F` --> ricerca utilizzando l'inverted index e BM25F come algoritmo di ranking
* `TFIDF` --> ricerca utilizzando l'inverted index e TF_IDF come algoritmo di ranking
* `doc2vec` --> ricerca e ranking utilizzando il modello doc2vec

Infine il parametro `--train` permette di allenare o indicizzareil modello e il parametro `-h` per richiamare l'help.
  
Utilizzando i modelli di Whoosh l'applicazione accetta query di tipo:
* multiple word query
* phrase query
* boolean query
* ranges query

## QUERY
Si mostra un elenco di query di esempio per mostrare le funzionalità del query language del search engine:
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


Elenco di query utilizzate per eseguire i **benchmark**:

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

## Benchmark

Per eseguire i benchmark delle differnti versinoi del search engine si è deciso di utilizzare il DCG e il NDCG.
I risultati sono consultabili nel file [benchmarks.pdf](benchmarks.pdf), oppure visualizzando il seguente [google sheet](https://docs.google.com/spreadsheets/d/184XW-nMnwe-zerCpLHr3Hv3OT4zajAOKnY5VZjWIOJU/edit?usp=sharing)
