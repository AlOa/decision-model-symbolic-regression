# Costruire equazioni un passo alla volta

## Regressione simbolica guidata da modelli decisionali e Monte Carlo Tree Search

*Un prototipo di ricerca che usa Ollama, Python e uv per esplorare una domanda semplice: può un modello decisionale locale suggerire all'algoritmo quale operatore aggiungere, mentre la ricerca si occupa di verificare sui dati se quel suggerimento funziona?*

---

## Sommario

La regressione simbolica cerca un'espressione matematica che descriva un insieme di dati. È un problema affascinante perché il risultato non è solo una previsione, ma una formula leggibile: una relazione che possiamo ispezionare, discutere e riusare.

Il prototipo descritto in questo articolo esplora un'idea precisa: invece di affidare tutta la ricerca a una procedura evolutiva o a una ricerca cieca, si può usare un modello decisionale locale come **bussola semantica**. A ogni passo il modello riceve una fotografia compatta dei dati, dell'espressione costruita fino a quel momento e del residuo; poi sceglie tra gli operatori disponibili. Le probabilità restituite diventano i prior di una ricerca Monte Carlo Tree Search, che resta libera di esplorare, valutare e correggere il suggerimento.

Il risultato è un prototipo completo: espressioni immutabili, dataset sintetici con rumore controllato, integrazione nativa con l'API decisionale di Ollama, ricerca PUCT, modalità greedy, CLI, artefatti riproducibili e una suite di test. Non è ancora un articolo sperimentale con risultati definitivi; è piuttosto il laboratorio che rende possibile quello studio.

---

## 1. Il problema

Un dataset numerico ci offre coppie di input e output. Vogliamo trovare una funzione che li spieghi. La regressione classica sceglie una famiglia di modelli e ne stima i parametri; la regressione simbolica va oltre e cerca la **struttura** della funzione: somme, prodotti, seni, esponenziali, rapporti.

Il spazio delle espressioni possibili è enorme. Anche con pochi operatori, il numero di alberi cresce in fretta. Le strategie tradizionali — programmazione genetica, ricerca esaustiva, metodi gradient-based su forme parametriche — funzionano, ma spesso trattano ogni operatore come un simbolo privo di significato rispetto al problema concreto.

Qui nasce l'ipotesi del progetto: e se il modello decisionale sapesse riconoscere, almeno in parte, che un residuo oscillante suggerisce una funzione trigonometrica? O che un residuo che cresce rapidamente suggerisce un esponenziale? Non serve che il modello risolva il problema; basta che proponga una distribuzione di probabilità utile a orientare la ricerca.

---

## 2. L'idea: il modello come bussola, la ricerca come motore

Il sistema separa nettamente due responsabilità.

Il **modello decisionale** non costruisce l'equazione. Riceve un contesto compatto e risponde a una domanda a scelta multipla: *quale operatore sarebbe utile aggiungere adesso?* Le alternative sono operatori come `add`, `mul`, `sin`, `exp`, `log`. La risposta contiene una probabilità per ogni alternativa e un valore di confidence.

La **ricerca** non si fida ciecamente del modello. Usa le probabilità come prior, genera applicazioni concrete, le valuta sui dati e usa la bontà del fit come ricompensa. In questo modo il modello suggerisce la direzione, ma sono i dati a decidere se una strada è buona.

Questa divisione è il cuore del progetto. Il modello è una bussola; la ricerca è il motore. La bussola può sbagliare, ma il motore ha la possibilità di correggere la rotta.

---

## 3. Come è fatto il sistema

### 3.1 La pipeline

Il flusso di un esperimento è lineare:

```text
dataset
   │
   ▼
contesto compatto
   │
   ▼
Ollama /v1/systemone
   │
   ▼
probabilità degli operatori
   │
   ▼
MCTS: espansione, valutazione, reward, backpropagation
   │
   ▼
migliore espressione + metriche + trace
```

Il dataset può venire da una suite di funzioni ground truth oppure da un file CSV. Il contesto viene costruito a ogni decisione. La ricerca mantiene un albero di espressioni candidate e un incumbent globale, cioè la migliore espressione valida incontrata fino a quel momento.

### 3.2 Le espressioni

Le espressioni sono alberi immutabili. Ogni nodo è una variabile, una costante, un operatore unario o un operatore binario. L'immutabilità non è un dettaglio estetico: permette di condividere sottoalberi tra i nodi dell'albero di ricerca senza rischi di mutazioni accidentali.

Gli operatori disponibili sono:

- binari: `+`, `-`, `*`, `/`;
- unari: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`.

Ogni operatore ha una descrizione leggibile, usata nei criteri della domanda al modello, e una guardia di dominio. La divisione per zero, il logaritmo di un numero non positivo, la radice di un numero negativo e l'esponenziale che va in overflow vengono intercettati. Un'espressione che produce `NaN` o `inf` non viene mai considerata valida.

La valutazione è vettoriale: l'espressione viene applicata a tutte le righe del dataset in una volta sola. La complessità è il numero di nodi, usato nei tie-break e negli artefatti.

### 3.3 Il contesto che vede il modello

Un modello decisionale locale non ha una finestra di contesto infinita. Per questo il sistema non invia l'intero dataset, ma una fotografia compatta:

- il passo corrente;
- l'espressione in forma testuale;
- un campione delle feature e del target;
- un riassunto del residuo: minimo, massimo, media, deviazione standard e alcuni valori campionati;
- una breve storia degli ultimi operatori applicati.

I valori sono arrotondati, il numero di righe è limitato e le chiavi sono serializzate in ordine fisso. Se il residuo non è calcolabile, il contesto lo dichiara esplicitamente invece di inviare numeri non finiti.

Questo è un punto delicato: il modello non vede tutto, e non deve. Deve vedere abbastanza per farsi un'idea della forma del residuo, non abbastanza per trasformare la decisione in un problema di lettura dei dati.

---

## 4. Il modello decisionale

L'integrazione con Ollama usa l'endpoint nativo `POST /v1/systemone`, non la chat generica. La differenza è importante: l'endpoint decisionale restituisce direttamente una distribuzione di probabilità sugli operatori, insieme a una confidence. Non serve estrarre numeri da un testo generato, non serve sperare che il modello rispetti un formato.

Prima di una run, il client verifica che il modello esista e che dichiari la capability `decision`. Se il modello non c'è, non è raggiungibile o non è decisionale, l'errore è esplicito e la CLI suggerisce di usare la modalità offline.

La richiesta contiene una sola domanda `choice`. I criteri sono gli operatori, con una descrizione sintetica. L'ordine dei candidati viene preservato. La risposta viene validata: l'operatore scelto deve appartenere ai candidati, ogni candidato deve avere una probabilità finita, la confidence deve essere un numero.

La selezione usa la probabilità più alta, non il campo `choice` in modo acritico. In caso di parità entro una tolleranza, vince l'ordine del catalogo. È una scelta di ingegneria: rende il comportamento ripetibile anche quando il modello è incerto.

Se la chiamata fallisce, la policy può restituire una distribuzione uniforme marcata esplicitamente come `fallback`. La ricerca continua, ma il trace registra l'errore. Non vengono mai spacciate probabilità inventate per probabilità del modello.

---

## 5. Gli operandi: ciò che il modello non sceglie

C'è una scelta di progetto che merita di essere raccontata. Il modello sceglie l'operatore, non l'operando. Se sceglie `mul`, la ricerca deve decidere *con cosa* moltiplicare l'espressione corrente.

Per farlo usa un **operand bank** deterministico: variabili, costanti, funzioni unarie delle variabili e alcuni termini composti come `2*x`, `x*x`, `sin(3*x)`, `exp(x/-5)`. Per ogni operatore binario il sistema prova entrambi gli orientamenti, `(current op operand)` e `(operand op current)`, valuta i risultati e tiene i migliori.

Questa separazione ha due vantaggi. Il primo è che il modello non deve scegliere tra centinaia di espressioni già formate: il suo spazio di scelta resta piccolo e leggibile. Il secondo è che la selezione degli operandi usa i dati, non il modello: la ricerca può scartare un'operazione semanticamente plausibile ma numericamente inutile.

---

## 6. La ricerca MCTS

La ricerca è un albero PUCT. Ogni nodo rappresenta un'espressione; ogni arco è un'operazione applicata. Il nodo conserva visite, reward accumulato, prior, figli e valutazione.

### 6.1 Selezione

Se un nodo ha figli non visitati, viene scelto quello con prior più alto. Altrimenti si massimizza:

```text
Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

dove `Q` è il reward medio, `P` è il prior del modello e `N` è il numero di visite. La costante `c_puct` regola l'esplorazione: valori alti spingono verso nodi poco visitati; valori bassi privilegiano quello che ha già funzionato.

### 6.2 Espansione

Per il nodo selezionato, il sistema chiede alla policy le probabilità degli operatori. Per ogni operatore genera le applicazioni valide, le valuta e tiene le migliori `top_applications`. Il prior dell'operatore viene distribuito tra le applicazioni mantenute e rinormalizzato su tutti i figli.

In questo modo il modello dice *quale tipo di mossa* ha senso, mentre i dati dicono *quale variante concreta* è migliore.

### 6.3 Rollout

Il rollout parte dal nodo appena espanso e simula una continuazione limitata. A ogni passo sceglie l'operatore con probabilità più alta che abbia un'applicazione valida, applica la migliore variante e valuta. Il rollout è quindi greedy: usa il prior del modello, non campiona casualmente.

### 6.4 Reward

La ricompensa di una valutazione valida è:

```text
reward = 1 / (1 + NMSE)
```

dove:

```text
NMSE = SSE / SST
```

Un fit perfetto dà reward `1`; un fit pessimo tende a `0`; un'espressione non valida vale `0`. La scelta di usare NMSE rende il reward confrontabile tra funzioni con scale diverse.

### 6.5 Backpropagation e incumbent

Dopo il rollout, il reward migliore viene propagato lungo il percorso fino alla radice. Il sistema mantiene anche un incumbent globale: la migliore espressione valida incontrata, non necessariamente quella più visitata dell'albero. A parità di reward vince l'espressione più semplice, poi l'ordine lessicografico.

### 6.6 Modalità greedy

La modalità `greedy` riproduce il ciclo più diretto: chiedi al modello l'operatore con la probabilità più alta, applica la migliore variante, calcola il residuo, ripeti. È utile come baseline, per demo rapide e per verificare il comportamento del modello senza la complessità dell'albero.

---

## 7. Dati, rumore e riproducibilità

La suite di test contiene undici funzioni ground truth distribuite su quattro livelli di complessità: costante/lineare, polinomiale, trigonometrico/esponenziale, composito/multivariato. Ogni funzione dichiara espressione, variabili, dominio e vincoli.

I dataset vengono generati campionando le feature nel dominio e aggiungendo rumore gaussiano al solo target:

```text
y = f(X) + ε,    ε ~ N(0, σ²)
```

Il rumore può essere fissato per deviazione standard oppure derivato da un rapporto segnale/rumore. Con rumore zero il target coincide esattamente con la funzione. Le feature non vengono mai perturbate. Lo stesso seed produce lo stesso dataset.

La riproducibilità è un requisito di progetto, non un accessorio. Il campionamento delle righe del contesto, l'ordine degli operatori, i tie-break, la cache delle decisioni e la policy offline sono tutti deterministici. Due run con stessa configurazione e stesso seed producono gli stessi artefatti, a meno di timestamp e durate.

---

## 8. Esecuzione e strumenti

Il progetto è gestito con `uv` e Python 3.12. I comandi principali sono:

```bash
uv sync
uv run pytest -q

uv run symbolic-regression run --function quadratic --policy offline \
  --iterations 20 --samples 64 --noise-std 0.05 --output runs --run-name demo

uv run symbolic-regression run --function sine --model d3:4b \
  --iterations 50 --output runs --run-name live

uv run symbolic-regression compare --function composite \
  --policies decision,uniform --iterations 50 --output runs --run-name compare
```

Ogni run scrive un `result.json`, un `trace.json` e un `summary.md`. Il risultato contiene l'espressione trovata (con la sua forma semplificata con SymPy), il ground truth quando disponibile, le metriche, la complessità, il numero di decision call, i fallback e la ragione di stop. Il trace contiene le decisioni del modello e le iterazioni della ricerca.

Il confronto produce anche `comparison.json` e `comparison.md`, con le due policy affiancate sullo stesso seed e sullo stesso budget.

La stessa funzionalità è disponibile come libreria Python, senza passare dagli artefatti:

```python
from symbolic_regression import fit

result = fit({"x": x}, y, operators=("mul", "add"), iterations=50)
print(result.simplified_expression)
print(result.to_dict())
```

`fit` accetta dati come mapping, array o data frame, un set di operatori opzionale e restituisce l'espressione trovata insieme alla diagnostica completa della ricerca.

La modalità offline permette di lavorare senza Ollama. Non è pensata per ottenere la migliore regressione possibile: è pensata per testare la pipeline, riprodurre esperimenti e fare demo in assenza di modelli locali.

---

## 9. Cosa abbiamo osservato

Il prototipo è completo e verificato. La suite di test conta oltre duecento casi; i test live contro `d3:4b` verificano che una richiesta `SystemOne` restituisca un operatore noto con una distribuzione completa e che un esperimento breve arrivi alla fine senza fallback.

Sono stati eseguiti esperimenti offline su un rappresentante per ogni livello di complessità, sia a rumore zero sia a rumore non zero. La riproducibilità è stata verificata confrontando due run sullo stesso dataset rumoroso: dopo aver rimosso timestamp e durate, result e trace sono identici.

È stato anche eseguito un confronto `decision` contro `uniform` su due livelli, con rumore non zero e budget minimo. Entrambe le policy hanno prodotto espressioni approssimate; con così poche iterazioni non è emersa una superiorità netta. Questo non è un risultato negativo: è il punto di partenza. Il prototipo serve proprio a rendere misurabile quella differenza quando il budget, il numero di seed e la varietà delle funzioni crescono.

In altre parole: abbiamo costruito il banco di laboratorio, non ancora l'esperimento completo.

---

## 10. Limiti

Il sistema ha limiti dichiarati.

Il primo è l'espressività del bank di operandi. La ricerca non può inventare un operando arbitrario; può comporre quelli disponibili. Espressioni molto composite possono richiedere più passi o non essere raggiungibili.

Il secondo è la qualità del modello. Il modello propone una distribuzione, ma non garantisce che sia buona. Con budget bassi, la differenza rispetto a una policy uniforme può essere minima.

Il terzo è l'assenza di un holdout. Le metriche sono calcolate sugli stessi dati usati per la ricerca. È un limite da affrontare prima di qualsiasi confronto sperimentale serio.

Ci sono poi limiti ingegneristici: le chiamate al modello sono sincrone, il contesto è compatto per necessità, l'API `SystemOne` è recente e il costo delle decisioni cresce con il numero di espansioni e rollout.

---

## 11. Prospettive

Le direzioni naturali sono chiare.

La prima è sperimentale: più funzioni, più seed, budget più alti, split di validazione e confronto sistematico tra policy. La seconda è strutturale: un bank di operandi più ricco o generato dinamicamente dal residuo. La terza è architetturale: chiamate asincrone o batch, caching su disco, potatura più aggressiva dell'albero.

Un'altra direzione interessante è il confronto tra modelli decisionali diversi. Il sistema è già progettato per cambiare modello con un flag; quello che manca è un protocollo di valutazione che dica quale modello, con quale contesto e con quale budget, produce i prior più utili.

---

## 12. Conclusioni

La regressione simbolica è un problema in cui la forma dell'ipotesi conta quanto i numeri. Il prototipo descritto qui prova a mettere un modello decisionale locale al servizio della ricerca, senza consegnargli il controllo.

Il modello propone; la ricerca verifica. Il contesto è compatto; il residuo è il filo che tiene insieme dati e struttura. Il reward è il giudice finale. È un'architettura semplice, ma abbastanza onesta da poter essere misurata.

Il prossimo passo non è aggiungere un altro pezzo di codice. È usare il banco di laboratorio per rispondere alla domanda iniziale: i prior di un modello decisionale aiutano davvero a trovare equazioni migliori, a parità di budget? Il prototipo è pronto per quella domanda.

---

## Per approfondire

- `docs/TECHNICAL.md`: esposizione tecnica dettagliata, modulo per modulo.
- `README.md`: installazione, comandi, configurazione e artefatti.
- `src/symbolic_regression/`: implementazione.
- `tests/`: suite di verifica.
