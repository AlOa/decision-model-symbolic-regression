# Esposizione tecnica — SimbolicRegression

Questo documento descrive come funziona il prototipo `SimbolicRegression`:
l'architettura, il modello dati, il formato delle espressioni, l'integrazione
con Ollama, l'algoritmo MCTS, la configurazione, gli artefatti e le verifiche.

Il documento riflette l'implementazione presente in `src/symbolic_regression/`.
Per la motivazione e i requisiti si rimanda alla change OpenSpec
`decision-model-symbolic-regression`.

---

## 1. Panoramica

Il sistema costruisce un'espressione simbolica in modo incrementale a partire
da un dataset numerico. A ogni passo:

1. lo stato corrente (dati campionati, espressione, residuo, storia) viene
   serializzato in un contesto compatto;
2. un modello decisionale Ollama riceve una domanda a scelta multipla in cui
   le alternative sono gli **operatori** disponibili;
3. le probabilità restituite dal modello diventano i **prior** di una ricerca
   Monte Carlo Tree Search;
4. la ricerca applica gli operatori all'espressione corrente, valuta i
   candidati sui dati e usa la bontà del fit come **reward**;
5. il residuo della migliore espressione trovata viene reinserito nel contesto
   del passo successivo.

In sintesi:

```text
dataset ──► contesto compatto ──► Ollama /v1/systemone
                                      │
                                      ▼
                              probabilità operatori
                                      │
                                      ▼
     espressione corrente ──► MCTS (PUCT + reward fit) ──► nuova espressione
                                      │
                                      ▼
                           residuo + metriche + trace
```

Il modello decisionale non sceglie direttamente l'espressione finale: fornisce
un prior semantico sugli operatori. La ricerca resta responsabile
dell'esplorazione, della scelta degli operandi e della selezione finale.

---

## 2. Pipeline di esecuzione

Il flusso completo di un esperimento è:

```text
CLI / RunConfig
      │
      ▼
load_dataset()
      │  suite sintetica o CSV
      ▼
make_policy()
      │  decision / uniform / offline
      ▼
MCTS(initial_expression = prima feature, dataset, policy, config)
      │
      ├─ selection (PUCT)
      ├─ expansion (operatori + operandi)
      ├─ rollout (policy greedy limitata)
      ├─ evaluation (MSE / NMSE / R² / reward)
      ├─ backpropagation
      └─ trace
      ▼
SearchResult
      │
      ▼
write_run_artifacts()
      ├─ result.json
      ├─ trace.json
      └─ summary.md
```

Per il confronto:

```text
compare_experiments()
      ├─ run_experiment(policy="decision")
      ├─ run_experiment(policy="uniform")
      └─ comparison.json + comparison.md
```

Tutte le policy usano lo stesso dataset, lo stesso seed e lo stesso budget.

---

## 3. Modello dati

Il contenitore principale è `Dataset` in `data.py`:

```python
Dataset(
    features={"x0": np.ndarray, "x1": np.ndarray},
    target=np.ndarray,
    metadata={...},
)
```

Validazioni eseguite alla costruzione:

- almeno una feature e almeno due righe;
- feature e target monodimensionali;
- stessa lunghezza per tutte le colonne;
- tutti i valori finiti;
- target numerico.

Il CSV viene caricato con `load_csv()`:

- colonna target esplicita;
- feature esplicite oppure tutte le altre colonne;
- delimitatore e presenza/assenza dell'header configurabili;
- errori con numero di riga e colonna per celle vuote, non numeriche o
  colonne mancanti.

I metadati registrano la provenienza (`function` o `csv`), l'espressione
ground truth quando disponibile, il dominio, il seed e la configurazione del
rumore.

---

## 4. Rappresentazione delle espressioni

Le espressioni sono alberi immutabili definiti in `expression.py`.

### 4.1 Nodi

```text
Variable(name)             variabile di input
Constant(value)            costante numerica finita
UnaryOp(name, operand)     operatore unario
BinaryOp(name, left, right) operatore binario
```

L'immutabilità è importante: ogni nodo dell'albero MCTS può condividere
sottoalberi senza rischi di mutazione durante la ricerca.

### 4.2 Operatori

| Categoria | Operatori |
|---|---|
| Binari | `add`, `sub`, `mul`, `div` |
| Unari | `sin`, `cos`, `exp`, `log`, `sqrt`, `abs` |

Ogni operatore è descritto da una `OperatorSpec` con:

- arietà;
- descrizione leggibile, usata nei criteri della domanda al modello;
- funzione NumPy;
- eventuale guardia di dominio.

Guardie principali:

- `div`: denominatore diverso da zero;
- `log`: argomento strettamente positivo;
- `sqrt`: argomento non negativo;
- `exp`: argomento inferiore a 700 per evitare overflow.

Se una guardia fallisce, o se il risultato contiene `NaN`/`inf`, l'espressione
viene considerata **non valida**. Non viene mai valutata parzialmente.

Il set di operatori è globale, ma ricerca e policy accettano un sottoinsieme
esplicito (`operators=...`). Il sottoinsieme viene validato, usato per limitare
le probabilità della policy, il bank degli operandi e le applicazioni ammesse
nell'albero.

### 4.3 Valutazione e formattazione

`evaluate(expr, features, n_rows)` valuta l'albero in modo vettoriale su tutte
le righe. Le espressioni costanti usano `n_rows` per conoscere la lunghezza
dell'output.

`to_string(expr)` produce una forma infissa con parentesi esplicite, ad esempio:

```text
((x * x) + sin((3 * x)))
```

La complessità è il numero di nodi (`node_count`). Questa metrica viene usata
nei tie-break e negli artefatti.

### 4.4 Semplificazione SymPy

Quando la ricerca produce un'espressione finale, `result.py` ne calcola una
forma semplificata con `simplify.py`:

- `to_sympy(expr)` converte l'albero nell'espressione SymPy equivalente; le
  variabili sono simboli reali e le costanti diventano razionali esatti
  (`0.5` → `1/2`);
- `simplify_expression(expr)` chiama `sympy.simplify` e restituisce la stringa
  risultante, oppure `None` se la conversione o la semplificazione falliscono.

La semplificazione è informativa: l'espressione originale e le metriche misurate
restano quelle della ricerca. Il risultato semplificato compare in
`SearchResult.simplified_expression`, in `result.json`, nel `summary.md`, nel
confronto e nell'output della CLI quando differisce dalla forma originale.

---

## 5. Suite di funzioni e generazione dei dataset

La suite è in `experiments/functions.py`. Ogni voce è una `FunctionSpec` con:

- nome;
- espressione leggibile;
- funzione NumPy;
- variabili;
- livello di complessità;
- dominio predefinito;
- eventuali vincoli.

### 5.1 Funzioni disponibili

| Livello | Funzioni | Espressioni di esempio | Dominio predefinito |
|---|---|---|---|
| 1 | `constant`, `linear` | `3.5`, `2*x - 1` | `x ∈ [-3, 3]` |
| 2 | `quadratic`, `cubic` | `x**2 - 3*x + 2`, `x**3 - 2*x` | `x ∈ [-3, 3]` / `[-2, 2]` |
| 3 | `sine`, `damped_sine`, `exponential` | `sin(2*x)`, `exp(-x/5)*sin(3*x)`, `exp(x/3)` | `x ∈ [-π, π]` / `[-3, 3]` |
| 4 | `rational`, `log_abs`, `poly2d`, `composite` | `1/(1+x**2)`, `log(abs(x)+1)`, `x0**2 + x0*x1 + x1`, `x**2 + sin(3*x) + 0.5` | `x ∈ [-3, 3]` / `poly2d`: `[-2, 2]²` |

Le funzioni con potenziali singolarità usano forme protette
(`log(abs(x)+1)`, `1/(1+x**2)`), così il campionamento resta finito.

### 5.2 Generazione deterministica

`generate_dataset()` in `experiments/datasets.py`:

1. crea un `np.random.default_rng(seed)`;
2. campiona ogni feature uniformemente nel dominio;
3. valuta la funzione ground truth;
4. aggiunge rumore al solo target;
5. restituisce un `Dataset` con metadati completi.

Il rumore è gaussiano additivo:

```text
y = f(X) + ε,    ε ~ N(0, σ²)
```

Due modalità:

```text
σ = noise_std
σ = std(f(X)) / 10^(noise_snr_db / 20)
```

Proprietà:

- `noise_std = 0` riproduce esattamente `f(X)`;
- le feature non vengono mai perturbate;
- stesso seed e stessa configurazione producono lo stesso dataset;
- due livelli di rumore con lo stesso seed condividono le feature e differiscono
  solo nel target.

I metadati registrano `noise_mode` (`std` o `snr_db`), `noise_value`,
`noise_sigma`, funzione, dominio, numero di campioni e seed.

---

## 6. Contesto decisionale

Il modello non riceve l'intero dataset. `context.py` costruisce un contesto
compatto e deterministico (`DecisionContext`) pensato per la finestra di
contesto ridotta di un modello decisionale locale.

### 6.1 Schema

```json
{
  "schema_version": 1,
  "step": 0,
  "expression": "x",
  "features": {"x": [ ... ]},
  "target": [ ... ],
  "residual": {
    "available": true,
    "count": 24,
    "min": -1.2,
    "max": 4.3,
    "mean": 0.8,
    "std": 1.1,
    "values": [ ... ]
  },
  "history": [
    {"operator": "mul", "operand": "x", "orientation": "right",
     "prior": 0.5, "confidence": 0.91, "source": "ollama"}
  ]
}
```

### 6.2 Campionamento e formattazione

- Il numero massimo di righe inviate è `context_rows` (default 24).
- Se il dataset è più piccolo, vengono inviate tutte le righe.
- Altrimenti gli indici sono scelti senza reinserimento con
  `np.random.default_rng(seed + step)` e ordinati.
- I valori sono arrotondati a 4 decimali per ridurre i token.
- Il residuo è calcolato sull'intero dataset; vengono inviati riassunto
  (`min`, `max`, `mean`, `std`) e al massimo `residual_values` valori campionati
  (default 16).
- Se la valutazione non è valida, `residual.available` è `false` e non viene
  serializzato alcun numero non finito.
- La storia contiene gli ultimi 8 passi del percorso corrente.

`to_json()` usa chiavi in ordine fisso, separatori compatti e
`allow_nan=False`. Questo rende il contesto adatto sia al modello sia al
caching e alla riproducibilità.

---

## 7. Integrazione con Ollama SystemOne

Il client è in `decision/client.py`. Non usa la chat generica: usa l'endpoint
decisionale nativo di Ollama.

### 7.1 Readiness

Prima di una run con policy `decision`:

1. `GET /api/tags` per elencare i modelli;
2. ricerca del modello per `name` o `model`;
3. lettura delle `capabilities`;
4. se assenti, `POST /api/show`;
5. verifica della capability `decision`.

Se il modello non esiste, non è raggiungibile o non è decisionale, viene
sollevato un errore tipizzato con un messaggio operativo. La CLI suggerisce di
usare `--policy offline`.

### 7.2 Richiesta

`POST /v1/systemone` con:

```json
{
  "model": "d3:4b",
  "state": { "...contesto compatto..." },
  "questions": {
    "operator": {
      "type": "choice",
      "instructions": "Choose the operator ...",
      "criteria": {
        "add": "addition",
        "mul": "multiplication",
        "sin": "sine"
      }
    }
  }
}
```

Caratteristiche:

- una sola domanda `choice` per decisione;
- l'ordine dei candidati è preservato;
- i criteri sono le descrizioni degli operatori;
- il modello risponde con il codice dell'operatore scelto e una distribuzione
  di probabilità.

### 7.3 Risposta

Esempio:

```json
{
  "model": "d3:4b",
  "answers": {
    "operator": {
      "type": "choice",
      "choice": "sin",
      "probabilities": {"sin": 0.7, "cos": 0.3},
      "confidence": 0.9
    }
  },
  "usage": {"input_tokens": 10, "output_tokens": 0}
}
```

Il client valida:

- presenza dell'oggetto `answers`;
- presenza della risposta della domanda;
- tipo `choice`;
- operatore scelto appartenente ai candidati;
- probabilità finite per **tutti** i candidati;
- confidence finita, se presente.

Risposte malformate, timeout, errori HTTP e modelli non decisionali producono
eccezioni tipizzate.

### 7.4 Retry, timeout e caching

- timeout configurabile (`request_timeout`, default 30 s);
- retry configurabili (`request_retries`, default 1);
- nessuna credenziale viene registrata;
- `OllamaOperatorPolicy` mantiene una cache in memoria indicizzata da un hash
  SHA-256 di modello, stato e criteri;
- una cache hit restituisce la stessa distribuzione, utile per riproducibilità
  e per evitare chiamate ripetute durante i rollout.

---

## 8. Policy degli operatori

Le policy sono in `decision/policy.py` e implementano tutte la stessa
interfaccia:

```python
probabilities(context: DecisionContext) -> PolicyResult
```

`PolicyResult` contiene:

- `probabilities`: distribuzione normalizzata sugli operatori;
- `selected`: operatore con probabilità più alta;
- `confidence`: confidence riportata dal modello;
- `source`: `ollama`, `cache`, `uniform`, `offline` o `fallback`;
- `model`;
- `error`;
- `elapsed_seconds`.

### 8.1 Scelta dell'operatore

La selezione usa la probabilità più alta, non direttamente il campo `choice`
del modello. In caso di parità entro una tolleranza (default `1e-9`), vince
l'ordine del catalogo. Questo rende la scelta deterministica.

Il valore `confidence` viene registrato nel trace e usato come diagnostica, ma
non sostituisce la distribuzione normalizzata.

### 8.2 Fallback

Se la chiamata fallisce e `allow_fallback=True`:

- viene restituita una distribuzione uniforme;
- `source="fallback"`;
- `error` contiene la causa;
- la ricerca continua;
- il numero di fallback viene conteggiato in `fallback_calls`.

Le probabilità di fallback non vengono presentate come output del modello:
sono esplicitamente marcate.

### 8.3 Policy uniforme e offline

- `UniformOperatorPolicy`: probabilità uguali per tutti gli operatori; usata come
  baseline di confronto.
- `OfflineOperatorPolicy`: distribuzione pseudo-casuale deterministica derivata
  da SHA-256 di seed, step, espressione e nome operatore. Non contatta Ollama e
  serve per test, demo e riproducibilità.

---

## 9. Operand bank

Il modello sceglie solo l'operatore. Gli operandi delle applicazioni binarie
sono generati da un **operand bank** deterministico (`search/operands.py`).

Il bank contiene:

- tutte le variabili;
- costanti predefinite: `-2, -1, 0, 0.5, 1, 2, 3, 3.5`;
- funzioni unarie applicate alle variabili: `sin(x)`, `cos(x)`, `exp(x)`,
  `log(x)`, `sqrt(x)`, `abs(x)`;
- termini composti quali `2*x`, `3*x`, `0.5*x`, `-1*x`, `5*x`, `x/2`, `x/5`,
  `x/-5`, `x*x`, `x+1`, `x-1`, `x+0.5`;
- per dataset multivariati, prodotti, somme e differenze tra coppie di
  variabili;
- applicazioni unarie ai termini composti, ad esempio `sin(3*x)`,
  `exp(x/-5)`.

Il bank viene deduplicato per stringa, ordinato per numero di nodi e poi per
stringa, e limitato a `max_operands` (default 256). La dimensione resta
controllata perché il modello non deve mai vedere gli operandi: è la ricerca a
valutarli sui dati.

Quando il set di operatori è ristretto, `build_operand_bank` riceve
`unary_names` e `binary_names`: i termini composti vengono creati solo se
l'operatore binario necessario è ammesso, e le applicazioni unarie solo per gli
operatori unari selezionati.

---

## 10. Monte Carlo Tree Search

L'algoritmo è in `search/mcts.py`.

### 10.1 Configurazione

| Parametro | Default | Significato |
|---|---|---|
| `iterations` | 50 | iterazioni massime |
| `max_nodes` | 500 | nodi massimi nell'albero |
| `max_depth` | 6 | profondità massima dell'espressione |
| `rollout_depth` | 3 | passi simulati da un nodo espanso |
| `c_puct` | 1.4 | costante di esplorazione |
| `top_applications` | 2 | applicazioni di operando mantenute per operatore |
| `max_operands` | 256 | dimensione del bank |
| `fit_threshold` | 0.995 | R² che ferma la ricerca |
| `max_steps` | 12 | limite sui passi dell'incumbent |
| `time_limit_seconds` | 300 | budget wall-clock |
| `greedy` | false | segue un solo percorso |
| `operators` | tutti | sottoinsieme di operatori ammessi |
| `context_rows` | 24 | righe campionate nel contesto |
| `residual_values` | 16 | residui campionati nel contesto |

### 10.2 Nodi e azioni

Un `Node` contiene:

- espressione;
- padre;
- azione che lo ha generato;
- profondità;
- figli;
- prior `P`;
- visite `N`;
- reward accumulato `W`;
- valutazione;
- flag `terminal` / `expanded`;
- risultato della policy che lo ha generato.

Un'`Action` contiene:

- operatore;
- nuova espressione;
- descrizione dell'operando (se binaria);
- orientamento (`right`, `left` o `unary`).

### 10.3 Selezione

Se un nodo ha figli non ancora visitati, viene scelto quello con prior più alto.
Altrimenti si massimizza il bound PUCT:

```text
Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

dove:

```text
Q(child) = W(child) / N(child)
```

La radice quadrata usa `max(1, N(parent))`. Le parità sono risolte in modo
deterministico dall'ordine dei figli.

### 10.4 Espansione

Per il nodo selezionato:

1. viene costruito il contesto compatto;
2. viene chiesta alla policy la distribuzione sugli operatori;
3. per ogni operatore con probabilità maggiore di zero:
   - **unario**: si applica l'operatore all'espressione corrente;
   - **binario**: si enumerano tutti gli operandi del bank, in entrambi gli
     orientamenti `(current op operand)` e `(operand op current)`;
4. le applicazioni non valide o neutre vengono scartate;
5. le applicazioni valide vengono valutate e ordinate per
   `(-reward, numero di nodi, stringa)`;
6. si tengono le migliori `top_applications` per operatore;
7. il prior dell'operatore viene distribuito tra le applicazioni mantenute e
   rinormalizzato su tutti i figli.

Applicazioni neutre tipiche vengono saltate: `+0`, `-0`, `*1`, `/1`, `*0`.

Se nessun operatore produce figli validi, il nodo diventa terminale.

### 10.5 Rollout

Il rollout parte dal figlio appena espanso (o dal nodo selezionato) e per al
massimo `rollout_depth` passi:

1. chiede le probabilità alla policy;
2. ordina gli operatori per probabilità decrescente;
3. sceglie il primo operatore che abbia un'applicazione valida;
4. applica la migliore applicazione per quell'operatore;
5. valuta e aggiorna il migliore locale;
6. si ferma prima se raggiunge `fit_threshold`.

Il rollout è quindi **greedy**: usa il prior del modello, non ricorre a
campionamento casuale.

### 10.6 Reward e incumbent

La reward di una valutazione valida è:

```text
reward = 1 / (1 + NMSE)
```

dove:

```text
NMSE = SSE / SST
SSE  = Σ (prediction - target)²
SST  = Σ (target - mean(target))²
R²   = 1 - NMSE
```

Proprietà:

- fit perfetto → `NMSE = 0` → `reward = 1`;
- fit pessimo → reward tendente a `0`;
- valutazione non valida → reward `0`;
- target costante con errore non nullo → `NMSE = ∞`, reward `0`, ma
  l'espressione resta formalmente valutabile.

L'incumbent globale è il miglior nodo valido incontrato. Il confronto usa:

1. reward più alta;
2. a parità, numero di nodi più basso;
3. a parità, ordine lessicografico della stringa dell'espressione.

### 10.7 Backpropagation

Dopo il rollout, la reward migliore del rollout viene aggiunta a ogni nodo del
percorso che va dal nodo di partenza del rollout alla radice:

```text
N ← N + 1
W ← W + reward
```

### 10.8 Terminazione

La ricerca si ferma quando:

- il miglior R² raggiunge `fit_threshold`;
- il miglior incumbent raggiunge `max_steps` passi;
- la radice non ha applicazioni valide (`no_valid_operator`);
- si esauriscono iterazioni, nodi o tempo (`budget_exhausted`).

Nota pratica: `max_depth` limita la crescita dell'albero, mentre `max_steps`
limita la profondità dell'incumbent. Con i default (`max_depth=6`,
`max_steps=12`) è normalmente `max_depth` a fermare la crescita; per rendere
`max_steps` il limite effettivo basta impostarlo a un valore minore o uguale a
`max_depth`.

### 10.9 Modalità greedy

Con `greedy=true` non viene costruito un albero:

1. si parte dall'espressione iniziale;
2. si chiede la distribuzione alla policy;
3. si sceglie l'operatore con probabilità più alta che abbia un'applicazione
   valida;
4. si applica la migliore applicazione per quell'operatore;
5. si ripete fino a `max_steps`, `max_depth`, fit threshold, tempo o nodi.

Questa modalità riproduce il ciclo "scegli l'operatore con la maggiore
confidenza e aggiungi il residuo" descritto nella richiesta originale. È utile
come baseline e per demo rapide.

### 10.10 Trace

Per ogni iterazione viene registrato:

- numero di iterazione;
- percorso selezionato (espressioni);
- operatore e operando espansi;
- orientamento;
- reward del rollout;
- passi del rollout;
- migliore espressione e reward correnti;
- sorgente della policy;
- numero di decision call e fallback.

Per ogni decisione viene registrato:

- step e espressione;
- candidati;
- probabilità;
- operatore selezionato;
- confidence;
- sorgente (`ollama`, `cache`, `uniform`, `offline`, `fallback`);
- errore, se presente;
- tempo trascorso.

---

## 11. Orchestrazione, configurazione e CLI

### 11.1 `RunConfig`

`config.py` definisce `RunConfig`, che contiene:

- dataset: `function` oppure `csv`, `target`, `features`, `delimiter`, `header`;
- modello: `model`, `base_url`, timeout e retry;
- policy: `decision`, `uniform`, `offline`;
- operatori: `operators` (sottoinsieme opzionale);
- dati sintetici: `samples`, `noise_std`, `noise_snr_db`, `seed`;
- ricerca: iterazioni, nodi, profondità, rollout, `c_puct`, top applicazioni,
  bank, soglia di fit, passi, tempo, `greedy`;
- output: `output`, `run_name`;
- contesto: `context_rows`, `residual_values`.

La configurazione viene risolta con precedenza:

```text
default < file TOML < argomenti CLI
```

Le chiavi sconosciute e i valori invalidi vengono rifiutati. Se non viene
specificato né `function` né `csv`, viene usata `quadratic`.

### 11.2 Runner

`experiments/runner.py`:

- `load_dataset()`: genera dalla suite o carica il CSV;
- `make_policy()`: crea la policy e, per `decision`, verifica la readiness del
  modello;
- `make_mcts_config()`: converte `RunConfig` in `MCTSConfig`;
- `run_experiment()`: esegue una run e scrive gli artefatti;
- `compare_experiments()`: esegue le policy richieste con lo stesso seed e
  budget e scrive il confronto;
- `format_outcome()`: riassunto leggibile per la CLI.

### 11.3 CLI

```bash
uv run symbolic-regression run [opzioni]
uv run symbolic-regression compare [opzioni]
```

Esempi:

```bash
uv run symbolic-regression run --function quadratic --policy offline \
  --iterations 20 --samples 64 --noise-std 0.05 --output runs --run-name demo

uv run symbolic-regression run --function sine --model d3:4b \
  --iterations 50 --output runs --run-name live

uv run symbolic-regression compare --function composite \
  --policies decision,uniform --iterations 50 --output runs --run-name compare
```

Opzioni principali:

```text
--function / --csv / --target / --features / --delimiter / --header
--model / --base-url / --policy / --policies
--seed / --samples / --noise-std / --noise-snr-db
--operators
--iterations / --max-nodes / --max-depth / --rollout-depth / --c-puct
--top-applications / --max-operands / --fit-threshold / --max-steps
--time-limit / --greedy / --request-timeout / --request-retries
--output / --run-name / --context-rows / --residual-values
--config
```

La CLI restituisce `0` in caso di successo e `2` in caso di configurazione o
runtime error, con messaggio su stderr.

### 11.4 Libreria Python

`api.py` espone l'interfaccia programmatica usata da un altro programma:

```python
from symbolic_regression import fit

result = fit(
    {"x": x},
    y,
    operators=("mul", "add", "sub"),
    iterations=50,
    seed=0,
)
print(result.simplified_expression, result.r2)
print(result.to_dict())
```

`fit(features, target, *, operators=None, policy="offline", ...)`:

- accetta un mapping nome → valori, un array 2-D con `feature_names`, o un
  oggetto simile a un data frame (`columns` più indicizzazione per colonna);
- accetta il target come array oppure come nome di colonna quando il contenitore
  lo include già;
- valida e filtra il set di operatori con lo stesso registro di `expression.py`;
- costruisce la policy richiesta (`offline`, `uniform`, `decision` o
  un'istanza custom) e la configurazione MCTS;
- non scrive artefatti: restituisce direttamente un `FitResult`.

`FitResult` contiene l'albero (`expression`), le forme testuali
(`expression_str`, `simplified_expression`), la `evaluation` completa, le
metriche derivate (`mse`, `nmse`, `r2`, `valid`), reward, stop reason, conteggi
di decisioni e fallback, durata, feature, operatori, history e record delle
decisioni. `to_dict()` serializza tutto (con `include_trace=False` omette
history e decisioni), `summary()` produce un riepilogo testuale e
`predict(features)` valuta l'espressione su nuovi dati.

`SymbolicRegressor` è un wrapper con stato minimo: le opzioni di ricerca passate
al costruttore vengono riusate da `fit()`, e gli attributi `expression_`,
`simplified_expression_`, `metrics_`, `diagnostics_` e `predict()` inoltrano al
`FitResult` corrente.

---

## 12. Artefatti

### 12.1 `result.json`

Contiene:

- `run`: modalità, policy, modello, seed, funzione/CSV, output, timestamp,
  durata;
- `search`: parametri di ricerca usati;
- `dataset`: righe, feature e metadati (inclusi rumore e ground truth);
- `result`: espressione migliore, sua forma semplificata con SymPy, metriche,
  reward, complessità, decision call, fallback, iterazioni, stop reason, durata.

### 12.2 `trace.json`

Contiene:

- `decisions`: record per ogni decisione;
- `iterations`: record per ogni iterazione;
- `stop_reason`.

### 12.3 `summary.md`

Riassunto leggibile con funzione, policy, espressione (e forma semplificata se
diversa), ground truth, MSE, NMSE, R², complessità, decision call, fallback,
iterazioni, stop reason, tempo, seed e rumore.

### 12.4 Confronto

`compare` produce:

- `comparison.json`: risultati per policy, dataset e metadati;
- `comparison.md`: tabella con espressione, R², NMSE, MSE, complessità,
  decisioni e stop reason per policy.

---

## 13. Riproducibilità

La riproducibilità è garantita da:

- seed esplicito per campionamento, rumore e policy offline;
- campionamento deterministico delle righe del contesto;
- ordinamento stabile di operatori, operandi e figli;
- tie-break per ordine di catalogo e stringa dell'espressione;
- cache delle risposte del modello per stato;
- esclusione di timestamp e durate dal confronto tra artefatti.

Nelle verifiche, due run offline con lo stesso seed e la stessa configurazione
hanno prodotto `result.json` e `trace.json` identici dopo la rimozione di
`created_at` e `elapsed_seconds`.

La cache del modello è importante anche per la riproducibilità quando il
modello è campionato: lo stesso contesto riutilizza la stessa distribuzione.

---

## 14. Test e verifiche

La suite è in `tests/`:

| File | Copertura |
|---|---|
| `test_expression.py` | AST, arietà, guardie, formattazione, valutazione |
| `test_data.py` | validazione dataset, CSV, errori |
| `test_metrics.py` | MSE, NMSE, R², reward, invalidità |
| `test_context.py` | schema, determinismo, residuo non finito |
| `test_functions.py` | tier, metadati, domini, campioni finiti |
| `test_datasets.py` | rumore, seed, SNR, zero noise |
| `test_client.py` | readiness, SystemOne, errori HTTP, parsing |
| `test_policy.py` | prior, tie-break, fallback, cache, offline |
| `test_mcts.py` | PUCT, espansione, rollout, reward, budget, greedy |
| `test_config.py` | TOML, override, validazione |
| `test_runner.py` | run, artefatti, confronto, CSV, riproducibilità |
| `test_cli.py` | comandi, help, exit code, config |
| `test_api.py` | `fit`, input, operatori, diagnostica, predict, wrapper |
| `test_simplify.py` | conversione SymPy e semplificazione |
| `test_live_smoke.py` | test live opzionali con Ollama |

Risultati al momento della stesura:

```text
uv run pytest -q
241 passed, 3 skipped
```

I tre test live sono saltati di default. Con `OLLAMA_LIVE=1`:

```bash
OLLAMA_LIVE=1 uv run pytest tests/test_live_smoke.py -q
# 3 passed
```

Sono stati inoltre eseguiti:

- run offline su un rappresentante per ogni tier, a rumore zero e non zero;
- verifica di riproducibilità su dataset rumoroso;
- confronto `decision` vs `uniform` su `quadratic` (tier 2) e `sine` (tier 3)
  con rumore non zero, stesso seed e stesso budget.

---

## 15. Limiti noti

- **Espressività del bank**: la ricerca non può generare operandi arbitrari;
  il bank è volutamente finito. Espressioni molto composite possono richiedere
  più passi o non essere raggiungibili.
- **Modello come prior**: la qualità del modello non è garantita. Con budget
  molto bassi, la differenza rispetto alla policy uniforme può essere minima.
- **Nessun holdout**: le metriche sono calcolate sugli stessi dati usati per la
  ricerca. È un limite dichiarato del prototipo.
- **Costo delle chiamate**: ogni espansione e rollout può chiamare il modello.
  La cache riduce le chiamate, ma budget alti restano costosi.
- **Contesto limitato**: il contesto è compatto per rientrare nella finestra
  del modello; non vede l'intero dataset.
- **Dipendenza dall'API**: `POST /v1/systemone` è un'API recente; il client
  isola il formato e fallisce in modo esplicito se il modello non è
  decisionale.
- **`max_depth` vs `max_steps`**: con i default è `max_depth` a limitare la
  crescita; `max_steps` va impostato a un valore utile se lo si vuole come
  limite effettivo.
- **Nessuna parallelizzazione**: le chiamate al modello sono sincrone.

---

## 16. Estensioni possibili

- valutazione con holdout o cross-validation;
- ottimizzazione multi-obiettivo accuratezza/complessità;
- ampliamento del catalogo di operatori e del bank di operandi;
- generazione dinamica del bank in base al residuo;
- chiamate asincrone o batch a SystemOne;
- confronto tra più modelli decisionali;
- cache su disco delle decisioni;
- esportazione delle espressioni in formati esterni;
- supporto a dataset reali e benchmark pubblici;
- potatura più aggressiva dell'albero e prior appresi dai dati.

---

## 17. Mappa dei file

```text
SimbolicRegression/
├── main.py                         entry point alternativo
├── pyproject.toml                  dipendenze e console script
├── README.md                       guida rapida
├── docs/
│   └── TECHNICAL.md                questo documento
├── examples/                       esempi eseguibili della libreria
├── src/symbolic_regression/
│   ├── api.py                      API pubblica: fit, FitResult, SymbolicRegressor
│   ├── expression.py               AST, operatori, valutazione, formattazione
│   ├── data.py                     Dataset, validazione, CSV
│   ├── metrics.py                  MSE, NMSE, R², reward
│   ├── context.py                  contesto compatto per il modello
│   ├── result.py                   stop reason e SearchResult
│   ├── simplify.py                 semplificazione SymPy dell'espressione finale
│   ├── config.py                   RunConfig, TOML, merge CLI
│   ├── cli.py                      comandi run e compare
│   ├── decision/
│   │   ├── client.py               client Ollama SystemOne
│   │   └── policy.py               policy decision/uniform/offline
│   ├── search/
│   │   ├── operands.py             operand bank deterministico
│   │   └── mcts.py                 PUCT, espansione, rollout, trace
│   └── experiments/
│       ├── functions.py            suite di funzioni ground truth
│       ├── datasets.py             generazione dataset con rumore
│       ├── runner.py               orchestrazione run/compare
│       └── artifacts.py            result/trace/summary/comparison
└── tests/                          suite pytest
```

---

## 18. Riferimenti rapidi

```bash
# Setup
uv sync
uv run pytest -q

# Run offline
uv run symbolic-regression run --function quadratic --policy offline \
  --iterations 20 --samples 64 --noise-std 0.05 --output runs --run-name demo

# Run con modello decisionale
uv run symbolic-regression run --function sine --model d3:4b \
  --iterations 50 --output runs --run-name live

# Confronto
uv run symbolic-regression compare --function composite \
  --policies decision,uniform --iterations 50 --output runs --run-name compare

# Test live
OLLAMA_LIVE=1 uv run pytest tests/test_live_smoke.py -q
```
