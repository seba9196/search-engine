# search-engine

Progetto per il corso universitario di Gestione dell'Informazione

# Installazione

- Scaricare le dipendenze necessarie
    ```bash
    python -m venv env
    . ./env/bin/activate
    pip install -r requirements.txt
    ```

# Utilizzo
L'applicazione accetta query di tipo:
* multiple word query
* phrase query
* boolean query

La ricerca per ora viene effettuata solo nel contenuto dei file indicizzati.

# QUERY
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
