# MaterialAllocator_RL

Dieses Repository enthält die Simulationsumgebung, Baselines, Reinforcement-Learning-Modelle sowie Auswertungs- und Analysekomponenten für die Bachelorarbeit **„Reinforcement-Learning-Lösung für den Material Allocator von PSI“**.

Ziel des Projekts ist es, die Materialzuordnung als sequenzielles Entscheidungsproblem zu modellieren und zu untersuchen, ob wertbasierte Reinforcement-Learning-Verfahren wie **DQN** und **QR-DQN** in einem regelbehafteten Material-Allocation-Kontext mit einer bestehenden **Greedy-Baseline** konkurrieren oder diese hinsichtlich des Scores übertreffen können.

## Projektkontext

Im Mittelpunkt steht ein materialbezogenes Allokationsproblem aus dem Umfeld des **Material Allocators**. Ausgehend von realitätsnahen Material- und Auftragsdaten werden zulässige Material-Auftragsschritt-Kombinationen erzeugt, bewertet und anschließend entweder mit einer heuristischen Greedy-Strategie oder mit RL-Verfahren gelöst.

## Forschungsziel

Untersucht wird, ob Reinforcement Learning für ein reales, regelbehaftetes und industriell motiviertes Material-Allocation-Problem geeignet ist und unter Einhaltung fachlicher Regeln und Restriktionen Lösungen erzeugen kann, die mit heuristischen Verfahren vergleichbar sind oder diese übertreffen.

## Inhalt auf einen Blick

- `src/ma_rl/domain`  
  Domänenmodelle, Regelwerk und Scoring der Zuordnungen

- `src/ma_rl/data`  
  Laden und Aufbereitung von PSI-/JSON-/SQL-Daten sowie Erzeugung von Szenarien

- `src/ma_rl/matching`  
  Generierung zulässiger Matches zwischen Material und Auftragsschritten

- `src/ma_rl/envs`  
  Gymnasium-Umgebungen für Single- und Multi-Scenario-Training mit Action Masking

- `src/ma_rl/baselines`  
  Greedy-Solver als Vergleichsbasis

- `src/ma_rl/rl`  
  Trainings- und Evaluationsskripte für DQN und QR-DQN

- `src/ma_rl/analysis`  
  Benchmarks, Auswertungen, Kennzahlen und Visualisierungen

- `src/ma_rl/experiments`  
  Experimentdefinitionen und reproduzierbare Versuchsabläufe

## Daten und Ergebnisse

- `data/raw`  
  Rohdatenquellen bzw. Eingabedaten

- `data/scenarios`  
  Demo-, Trainings-, Validierungs- und Testszenarien

- `data/outputs`  
  Trainierte Modelle, Benchmark-Ergebnisse, Testset-Auswertungen und erzeugte Visualisierungen

## Verwendete Verfahren

In diesem Projekt werden drei zentrale Lösungsansätze miteinander verglichen:

- **Greedy-Baseline**  
  Heuristische Auswahl zulässiger Matches auf Basis lokaler Bewertungslogik

- **DQN (Deep Q-Network)**  
  Wertbasiertes Reinforcement-Learning-Verfahren für diskrete Entscheidungsprobleme

- **QR-DQN (Quantile Regression Deep Q-Network)**  
  Distributionelle Erweiterung von DQN mit Quantil-basierter Approximation zukünftiger Returns

## Methodische Grundidee

Das Material-Allocation-Problem wird als sequenzieller Entscheidungsprozess modelliert:

- **Zustand:** aktueller Stand der verfügbaren Materialien, offenen Auftragsschritte, zulässigen Matches und des bisherigen Assignment Sets
- **Aktion:** Auswahl eines zulässigen Matches
- **Reward:** scorebezogene Veränderung des aktuellen Assignment Sets
- **Ziel:** Konstruktion eines hochwertigen, konfliktfreien Assignment Sets unter Einhaltung fachlicher Regeln und Restriktionen

## Voraussetzungen

Für die Ausführung des Projekts werden in der Regel benötigt:

- Python `3.11`
- `pip`
- optional: virtuelle Umgebung mit `venv` oder `conda`

