# A2C auf Transportaktien — Update-Report (Stand 07.10.2026)

**Projekt:** Applied Reinforcement Learning (A2C) on Transportation Stocks — 2026 data update
**Autor:** Christopher Voizard, MatrNr 106225
**Zweck:** Fortschreibung des abgegebenen Notebooks mit Daten bis 06.10.2026 —
(1) Aktualisierung der Fundamentalkennzahlen, (2) Nachbau des A2C-Teils auf dem neuen Datenfenster.

> **Einordnung vorab:** Alle Ergebnisse dieses Reports stammen aus einem **Nachbau** in einer
> anderen Umgebung als das Original-Notebook. Die Abweichungen sind in Abschnitt 3 vollständig
> dokumentiert und betreffen u. a. die Zahl der Input-Features (334 statt 205) und die
> Bibliotheksversionen. Die Zahlen sind deshalb **nicht 1:1** mit dem Notebook vergleichbar,
> sondern als unabhängige Plausibilitätsprüfung zu lesen.

---

## 1. Kennzahlen-Update (Fundamentaldaten)

Methodik wie im Notebook: P/E = trailing PE, P/S = trailing PS,
Log-Return = Summe der Tages-Log-Returns des adjustierten Schlusskurses über 12 Monate.

| Kennzahl | Boeing (BA) | Airbus (AIR.PA) | Toyota (TM) |
|---|---|---|---|
| P/E Notebook (2022) | n/a (KeyError) | 19,887 | 10,375 |
| P/E heute | 70,79 | 25,04 | 7,91 |
| P/S Notebook (2022) | 1,502 | 1,458 | 0,0063 ⚠️ |
| P/S heute | 1,60 | 1,93 | 0,68 |
| 12M-Log-Return Notebook | −0,3025 | −0,1451 | −0,1486 |
| 12M-Log-Return heute | −0,1558 | −0,0945 | −0,0743 |
| Kurs seit 31.08.2022 | +20,5 % | +50–65 % (5J) | +39,5 % |

**Was sich dadurch an den Notebook-Aussagen ändert**

1. **P/E (§1.2) — bestätigt, teils verschärft.** Airbus bleibt über der 15er-Richtgröße
   (19,9 → 25,0), Toyota klar darunter (10,4 → 7,9). Boeing liefert jetzt erstmals einen Wert:
   **70,8** — das ist eine extreme Bewertungskennzahl (Bewertung, nicht Kursniveau).
2. **P/S (§1.3) — Datenfehler im Notebook aufgedeckt.** Toyota war nie „nahe null": der Wert
   0,0063 war ein yfinance-Ausreißer. Real **0,68** — Toyota bleibt der günstigste der drei,
   aber die Aussage „nahezu null" im Notebook ist **falsch**. Boeing 1,60, Airbus 1,93.
3. **Log-Returns (§1.4) — Kernaussage hält.** Alle drei sind auch im letzten Jahr negativ
   (BA −15,6 %, AIR −9,5 %, TM −7,4 %). Buy-and-Hold hätte auch diesmal verloren.
   **Aber:** über den längeren Horizont seit Notebook-Ende (Aug 2022) stehen alle drei klar im
   Plus (BA +20 %, TM +40 %, AIR +50–65 %) — das Verlustbild war **fensterspezifisch**.

---

## 2. A2C-Nachbau auf neuen Daten (Boeing)

**Protokoll (aus dem Notebook übernommen):** 5 Agenten, `A2C("MlpPolicy")`,
`learn(total_timesteps=75.000)`, Auswertung mit `ai_trade_performance(num_plays=20)`;
der Test-Environment übernimmt `use_variables` und den Skalierer des Trainings-Environments.
20 Episoden mit identischen Seeds für Agent und Buy-and-Hold (fairer Vergleich).

| | Zeitraum | Datenpunkte |
|---|---|---|
| Trainingsfenster (in-sample) | 06.10.2021 – 31.05.2024 | ~660 Handelstage |
| Testfenster (out-of-sample) | 01.06.2024 – 06.10.2026 | ~580 Handelstage |

### 2.1 Ergebnisse je Agent (End-NAV, Start = 1,0; Mittel über 20 Episoden)

| # | Seed | Train Agent | Train B&H | Agent besser | Test Agent | Test B&H | Agent besser |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 1,6705 | 0,9647 | 100 % | **0,5616** | 0,9475 | 0 % |
| 2 | 101 | 2,4073 | 0,9671 | 100 % | 1,2403 | 0,9408 | 100 % |
| 3 | 102 | 4,0534 | 0,9956 | 100 % | 0,9828 | 0,9412 | 75 % |
| 4 | 103 | 7,0580 | 0,9711 | 100 % | 1,0106 | 0,9493 | 70 % |
| 5 | 104 | 5,1547 | 0,9727 | 100 % | 1,1326 | 0,9485 | 80 % |
| **Ø** | | **4,0688** | **0,9742** | **100 %** | **0,9856** | **0,9455** | **65 %** |

### 2.2 Handelsverhalten (Aktionen über je eine Episode)

| # | Train short / cash / long | Test short / cash / long |
|---|---|---|
| 1 | 268 / 18 / 348 | 239 / 17 / 299 |
| 2 | 295 / 39 / 300 | 304 / 24 / 227 |
| 3 | 297 / 24 / 313 | 279 / 29 / 247 |
| 4 | 246 / 212 / 176 | 260 / 201 / 94 |
| 5 | 296 / 15 / 323 | 309 / 12 / 234 |

Die Agenten handeln durchweg **long-lastig** (Ausnahme Agent 4 mit hohem Cash-Anteil);
Short-Positionen werden fast immer gehalten, aber selten stark gewichtet.

### 2.3 Interpretation

- **In-sample (Trainingsfenster) schlägt der Agent Buy-and-Hold scheinbar deutlich**
  (Ø 4,07 vs. 0,97). Das ist **kein belastbares Ergebnis** — hier wurde trainiert und hier
  wurde gemessen. Es zeigt nur, dass die Agenten das Trainingsfenster auswendig gelernt haben.
- **Out-of-sample (Testfenster)** ist das Bild nüchtern: Ø Agent 0,986 vs. Ø B&H 0,946,
  Agent in **65 %** der Episoden besser — aber mit **riesiger Streuung**: Agent 1 verliert
  massiv (0,56), Agent 2 gewinnt deutlich (1,24). Zwei von fünf Agenten verlieren in ≥ 25 %
  der Episoden gegen Buy-and-Hold.
- **Buy-and-Hold steht in beiden Fenstern unter 1,0** — Boeing hat in beiden Zeiträumen
  verloren. Die Strategiefrage ist hier also „weniger verlieren", nicht „gewinnen".
- **Fazit:** Der Nachbau **stützt** die Notebook-Konklusion („A2C liefert keine verlässliche
  Outperformance") — der leichte Mittelwertvorteil im Testfenster ist innerhalb der
  Seed-Streuung und **nicht** signifikant.

**Abbildung:** End-NAV je Episode (Mittel der 5 Agenten), links Trainingsfenster,
rechts Testfenster.

![A2C auf Boeing (neue Daten)](figs/update_2026_a2c.png)

---

## 3. Methodik-Abweichungen gegenüber dem Original-Notebook

Der Nachbau lief **nicht** in Chris' conda-Env (`tensorflow`, Python 3.8.8, SB3 1.6.0,
gym 0.21), sondern in einem Linux-Container. Alle Abweichungen sind bewusst und dokumentiert:

| Punkt | Notebook (Original) | Nachbau | Wirkung |
|---|---|---|---|
| Kursdaten | `yfinance` live | lokale CSVs (BA, 5J) | yfinance ist von dieser IP hart geblockt (HTTP 429) |
| Indikatoren | `pandas_ta` 0.3.14b0 | `pandas_ta_classic` | **204 → 348 Indikatoren** ⇒ Observation **205 → 334** |
| Parallelisierung | `Pool(cpu_count())` | seriell (`_cores = 0`) | Container begrenzt Prozesse; Ergebnisse identisch |
| RL-Stack | SB3 1.6.0 + gym 0.21 | SB3 2.9.0 + gymnasium 1.4 | dünner Gymnasium-Adapter um das alte Env |
| Modell | 5 Agenten `agent1..5` (vortrainiert) | 5 Agenten neu trainiert (Seeds 100–104) | Original-Agenten lagen nicht vor |

**Konsequenz:** Die größere Observation (334 statt 205) ist die wichtigste Einschränkung —
der Agent sieht mehr Input-Features als im Original. Ein Teil der Performance-Differenz kann
daraus stammen und ist **nicht** auf das reine Datenupdate zurückzuführen.

**Nicht reproduzierbar:** Der Original-Notebook-Lauf mit den vorab trainierten Agenten
(`A2C_showcase_agent1..5`). Chris hat nur **einen** Agenten (`A2C_showcase_agent.zip`)
nachgeliefert; ein Vergleich „Original-Agent vs. neu trainierter Agent" war daher nicht möglich.

## 4. Artefakte

| Datei | Inhalt |
|---|---|
| `UPDATE_2026_REPORT.md` / `.pdf` | dieser Report |
| `figs/update_2026_a2c.png` | End-NAV je Episode, Train vs. Test |
| `figs/update_2026_1y.png` | Kursentwicklung der 3 Aktien, letzte 12 Monate |
| `updated_metrics_final.json` | aktualisierte Fundamentalkennzahlen |
| `aif_run/results_ba_newdata.json` | Roh-Ergebnisse der 5 Agenten (Train + Test) |
| `aif_run/agent_ba_1..5.zip` | die 5 neu trainierten A2C-Modelle |
| `aif_run/env_runner.py` | Env-Aufsatz (lokale Daten, pandas_ta-Alias, Seriell-Patch) |
| `aif_run/train_eval.py` | Training + Auswertung (Protokoll aus dem Notebook) |
| `aif_run/plot_results.py` | Tabellen-Ausgabe + Figur |
| `data/BA_5Y_daily.csv` | Boeing-Tagesdaten (5 Jahre) |
| `course/aif_environment.py`, `course/aif_analysis.py` | Chris' Kurs-Module (Laufzeit-Abhängigkeit) |

**Reproduzieren:**
```bash
cd aif-latex
python3 aif_run/train_eval.py      # trainiert 5 Agenten + wertet aus  (~35 min)
python3 aif_run/plot_results.py    # Tabelle + Figur
python3 aif_run/make_pdf.py UPDATE_2026_REPORT.md AIF_Update_2026.pdf
```
