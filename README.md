# MaterialAllocator_RL

Dieses Projekt untersucht die Materialzuordnung als Optimierungsproblem mit Reinforcement Learning. Ausgehend von realitaetsnahen Material- und Auftragsdaten werden zulaessige Material-Auftragsschritt-Kombinationen nach festen Regeln erzeugt, bewertet und anschliessend entweder mit der Greedy-Baseline oder mit RL-Modellen wie DQN und QR-DQN geloest.

## Inhalt auf einen Blick

- `src/ma_rl/domain`: Domaenenmodelle, Regelwerk und Scoring der Zuordnungen
- `src/ma_rl/data`: Laden von PSI-/JSON-/SQL-Daten und Erzeugung von Szenarien
- `src/ma_rl/matching`: Generierung zulaessiger Matches zwischen Material und Auftragsschritten
- `src/ma_rl/envs`: Gymnasium-Umgebungen fuer Single- und Multi-Scenario-Training mit Action-Mask
- `src/ma_rl/baselines`: Greedy-Solver als Vergleichsbasis
- `src/ma_rl/rl`: Trainings- und Evaluationsskripte fuer DQN und QR-DQN
- `src/ma_rl/analysis` und `src/ma_rl/experiments`: Benchmarks, Plots und Auswertungen

## Daten und Ergebnisse

Im Ordner `data/scenarios` liegen Demo-, Trainings-, Validierungs- und Testszenarien. `data/raw` enthaelt die Rohdatenquellen, und unter `data/outputs` befinden sich trainierte Modelle, Benchmark-Ergebnisse, Testset-Auswertungen und erzeugte Visualisierungen.
