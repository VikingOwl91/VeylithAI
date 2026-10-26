# veylith v2: Roadmap

Stand 2026-10-26. Evidenzbasis: `STATUS-2026-10-26.md`. Reihenfolge nach
Korrektheit vor Performance vor Architektur. Jede Massnahme braucht ein
Abnahmekriterium, das ohne riskante GPU-Experimente pruefbar ist.

Legende Aufwand: S = unter 1 Tag, M = 1 bis 2 Tage, L = mehr als 2 Tage.

## P0: Stabilitaet und Korrektheit

### P0-1 Trainingslauf-Integritaet: atomare Checkpoints, periodisches Speichern, Resume, minimale Testbasis

**Status: umgesetzt am 2026-10-26** (`--resume`, `--save-every`,
`--stop-after`, `--state-path`, `--device`, `veylith/v2/tests/test_train.py`).
Die deterministischen Eval-Fenster sind nach P0-2 verschoben: sie aendern die
Evaluationssemantik und waren nicht Teil der Freigabe.

- **Problem und Evidenz**: D1 (Checkpoint-Pfad ohne Verzeichnis stuerzt
  nach dem Training ab, auf CPU reproduziert) und D2 (kein Resume, kein
  periodischer Checkpoint). Dazu D4: es gibt keinen einzigen Test.
  Dokumentierte GPU-Faults machen verlorene Laeufe zum Regelfall, nicht
  zur Ausnahme.
- **Nutzen**: kein verlorener Lauf mehr; laengere Laeufe und Fortsetzung
  nach Absturz werden erst moeglich; Regressionen werden sichtbar.
- **Aufwand**: M, etwa 150 Zeilen Trainingsaenderung plus etwa 150 Zeilen
  Tests.
- **Risiken und Abhaengigkeiten**: Checkpoint-Format muss rueckwaerts-
  kompatibel bleiben (`weights_only=False`-Pfad, Tokenizer-Dicts).
  Exaktheit von Resume ist auf CPU beweisbar, auf GPU nur naeherungsweise
  (RNG-Zustand der GPU nicht Teil eines CPU-Tests). Keine Abhaengigkeit
  von GPU-Freigabe.
- **Abnahme**:
  1. `--checkpoint ckpt.pt` (ohne Verzeichnis) speichert erfolgreich.
  2. Schreiben ist atomar (temporaere Datei plus `os.replace`); ein
     abgebrochener Prozess hinterlaesst keinen halben Checkpoint.
  3. Ein 12-Iterationen-Lauf und ein 6-plus-Resume-6-Lauf ergeben auf CPU
     pro Iteration identische Loss-Werte.
  4. Der Checkpoint enthaelt Modell, Optimizer, Schritt, best_val,
     RNG-Zustaende und einen Config-Hash; Resume bricht bei Abweichung mit
     klarer Meldung ab.
  5. `python -m pytest -q` laeuft auf CPU gruen in unter 90 s und deckt
     Config-Invarianten, MTP-Verschiebung, Checkpoint-Roundtrip und
     Resume-Gleichheit ab.
  6. Der vorhandene `v2/checkpoints/ckpt.pt` laedt weiterhin und samplet.

### P0-2 Messwerte trennen und Evaluation reproduzierbar machen

**Status: umgesetzt am 2026-10-26** (`--eval-mode`, `--eval-seed`,
`--select-metric`, Komponenten im Log und in den Metadaten, feste
Eval-Fenster, Selektionsziel in Checkpoint und State).

- **Problem und Evidenz**: D3 und D6. Die gemeldete Loss ist eine
  Mischung (5.7144 = 3.1895 Haupt + 2.474 MTP + 0.051 Aux), MTP stellt
  43 %. Eval-Fenster sind zufaellig, also zwischen Laeufen nicht
  vergleichbar. Dazu W2: die `bits/char`-Spalte reproduziert nicht.
- **Nutzen**: v2 wird mit v1 und mit sich selbst vergleichbar; die
  Modellselektion optimiert die Groesse, die tatsaechlich interessiert.
- **Aufwand**: S, etwa 60 Zeilen.
- **Risiken und Abhaengigkeiten**: Aendert die Semantik von Early
  Stopping, wenn das Selektionsziel wechselt. Deshalb zuerst nur
  zusaetzlich loggen, Default unveraendert lassen, Wechsel separat
  entscheiden. Bestehende Checkpoints bleiben vergleichbar, weil deren
  Composite-Wert erhalten bleibt.
- **Abnahme**:
  1. Pro Eval werden Haupt-CE, MTP-CE und Aux getrennt geloggt.
  2. Ein Test rechnet die drei Werte aus dem Modell nach und vergleicht
     sie mit der Ausgabe.
  3. Eval nutzt einen festen, geseedeten Fenstersatz; zwei identische
     Laeufe liefern identische Eval-Zahlen.
  4. Selektionsziel ist explizit konfigurierbar, Default unveraendert.
  5. `bits/char` in den READMEs ist aus den Tabellenwerten nachrechenbar
     oder entfernt.

### P0-3 Doku-Widersprueche aufloesen

**Status: umgesetzt am 2026-10-26.** W1, W2 und W4 sind korrigiert, W3 ist
als offene Frage dokumentiert. Neu im Repo: `debug/phase4-scaling.md`. Die
unversionierte Arbeitskopie im Investigation-Worktree bleibt unangetastet;
massgeblich ist die versionierte Fassung.

- **Problem und Evidenz**: W1 und W4. Die Repo-Doku behauptet einen
  behobenen Fault und empfiehlt Sparse-Adoption, was die spaetere
  Untersuchung nicht deckt. `phase4-scaling.md` steht auf PENDING,
  obwohl die Ergebnisse existieren.
- **Nutzen**: keine Fehlentscheidung auf Basis veralteter Aussagen.
- **Aufwand**: S.
- **Risiken**: Die GPU-Untersuchung ist fremdes Terrain. Nur Aussagen
  korrigieren, keine Evidence umschreiben, keine neuen Experimente.
- **Abnahme**: READMEs nennen den Stand von 2.16.0.dev20261006 mit dem
  E16/K4-Fault, verweisen auf die Untersuchung und empfehlen Sparse nur
  unter Vorbehalt. Die PENDING-Stelle ist gefuellt oder entfernt.

### P0-4 Konfigurations- und Grenzfallpruefungen

**Status: umgesetzt am 2026-10-26** (`--device` validiert, `validate_config`
und `validate_args`, MTP-Guard, klares Fehlerbild fuer Prompts,
Dependency-Manifest, 29 zusaetzliche CPU-Tests).

- **Problem und Evidenz**: D7 (`block_size 1` plus MTP ergibt NaN) und
  D5 (kein `--device`). Beides still oder nur per Umgebungsvariable
  umgehbar.
- **Nutzen**: klare Fehler statt stiller NaNs; CPU-Verifikation ohne
  Trick.
- **Aufwand**: S.
- **Risiken**: keine, rein additiv.
- **Abnahme**: `--device auto|cpu|cuda|mps` mit Validierung; MTP wird bei
  zu kurzen Fenstern mit klarer Fehlermeldung abgelehnt; Prompts mit
  unbekannten Zeichen ergeben eine verstaendliche Meldung; ungueltige
  Argumente werden vor dem Training abgelehnt; Dependency-Manifest
  vorhanden; Tests decken alle Faelle ab.

## P1: Performance und Training

### P1-1 Ursache des Wall-Clock-Abstands messen, dann gezielt optimieren

**Status: Phase A (Instrumentierung) umgesetzt am 2026-10-26. Phase B
(GPU-Messung) ist vorbereitet, aber nicht gestartet; sie wartet auf eine
gesonderte Freigabe.**

- **Problem und Evidenz**: D8 und W3. Bei Batch 64 stehen 148.93 ms
  GPU-Step (110 010 tok/s) etwa 5.5 it/s Wall-Clock gegenueber. Datenladen
  ist gemessen 0.79 ms und damit nicht die Ursache. Kandidaten: amortisierte
  Evaluation, ein Synchronisationspunkt pro Iteration durch Clip, reine
  Host-Overhead-Anteile.
- **Nutzen**: 10 bis 25 % mehr Durchsatz ohne Modellaenderung, wenn sich
  die Ursache bestaetigt.
- **Aufwand**: S bis M, aber **GPU-Freigabe noetig** fuer eine begrenzte
  Messreihe (ein Profil-Lauf plus eine Wiederholung).
- **Risiken**: Keine Aussage ohne Messung. Neue Kernel-Pfade vermeiden,
  solange die hipBLASLt-Lage nicht geklaert ist.
- **Abnahme**: Profil zeigt die Anteile konkret; die Aenderung senkt die
  Wall-Clock um mindestens 10 % bei identischer Loss-Kurve; kein Fault.

### P1-2 Datenpfad, nur bei belegtem Bedarf

- **Problem und Evidenz**: Vektorisierte Batch-Erzeugung ist auf CPU 6.9x
  schneller (0.11 statt 0.79 ms), aber die absolute Ersparnis liegt unter
  1 % eines Steps.
- **Nutzen**: gering, nur relevant bei deutlich groesseren Batches oder
  wenn P1-1 Kopier-Stalls zeigt.
- **Aufwand**: S.
- **Risiken**: Kein Nutzen, wenn die Messung ihn nicht zeigt. Dann
  verwerfen statt einbauen.
- **Abnahme**: mindestens 5 % End-to-End-Gewinn, sonst nicht mergen.

### P1-3 Batch- und Blockgroessen systematisch vermessen

- **Problem und Evidenz**: Die vorhandene Messung deckt nur wenige
  Punkte ab, und die Analyse stammt aus einer anderen Fragestellung.
- **Nutzen**: belastbare Defaults fuer Batch und Block, Aussage ueber
  Speichergrenzen.
- **Aufwand**: S, **GPU-Freigabe noetig**.
- **Risiken**: OOM ist ein Messergebnis, kein Fehler, aber die Laeufe
  muessen begrenzt bleiben.
- **Abnahme**: Tabelle mit Batch 16/32/64/128 bei festem Token-Budget,
  je drei Bloecke, plus dokumentierter Speicherspitze.

### P1-4 Evaluationsprotokoll und Modellqualitaet

**Status: umgesetzt am 2026-10-26** (`veylith/evaluate.py`,
`veylith/evaluation/REPORT.md`, `veylith/tests/test_evaluate.py`).

- **Problem und Evidenz**: es gab kein gemeinsames Protokoll fuer v1 und v2;
  die gespeicherten Val-Losses sind wegen Tokenizer, Loss-Definition und
  Auswertungsfenstern nicht vergleichbar.
- **Ergebnis**: auf sechs gemeinsamen Val-Abschnitten (3067 Zeichen, main CE
  nur) v1-char 1.913, v1-bpe 1.922, v2-bpe 2.060 bits/char. Auf identischen
  BPE-Token ist v2 0.213 nats/token schlechter (paarweise, SE 0.038). Der
  Router ist gesund: alle Experten genutzt, Entropie 0.76-0.84 normiert,
  abschnittsabhaengige Auswahl. Die v2-Generierung ist weniger repetitiv,
  aber inkohaerenter als v1.
- **Nutzen**: reproduzierbare Entscheidungsgrundlage fuer P2.
- **Risiken**: sechs Abschnitte sind eine kleine Stichprobe; absolute Werte
  sind abschnittsspezifisch.
- **Abnahme**: erfuellt. Zwei unabhaengige Laeufe erzeugen identische
  `results.json`.

## P2: Architektur und Modellqualitaet

### P2-1 MTP-Gewicht und MTP-Kopf

- **Problem und Evidenz**: D11. MTP stellt 43 % der Loss; unklar, ob das
  die Haupt-CE verbessert oder bremst. Die kontrollierte Evaluation zeigt den
  v2-Checkpoint 0.138 bits/char schlechter als v1 bei gleichem Tokenizer, also
  besteht Klaerungsbedarf am Trainingsziel.
- **Nutzen**: moeglicher direkter Qualitaetsgewinn bei gleichem Budget.
- **Aufwand**: S bis M, **GPU-Freigabe noetig**.
- **Risiken**: Ohne festes Token-Budget sind Vergleiche wertlos; erst
  P0-2 umsetzen.
- **Abnahme**: drei Arme (0.0, 0.2, 0.5) bei identischem Token-Budget,
  Haupt-CE auf dem gemeinsamen Eval-Satz (`veylith/evaluate.py`), Streuung
  ueber zwei Seeds.

### P2-2 Kapazitaet gegen Datenmenge

- **Problem und Evidenz**: D12. 25.6M Parameter auf 0.5M Token; die
  Validierung dreht frueh nach oben. Kontrollierte Evaluation: v2 liegt trotz
  4.8-facher Parameterzahl 0.138 bits/char ueber v1 (paarweise, SE 0.025),
  bei gleichem Tokenizer.
- **Nutzen**: wahrscheinlich der groesste Qualitaetssprung, entweder
  durch mehr Daten oder durch ein kleineres Modell.
- **Aufwand**: M bis L, **GPU-Freigabe noetig**, plus Datenbeschaffung.
- **Risiken**: Neuer Datensatz aendert die Vergleichbarkeit mit v1 und
  v2. Nur mit klarer Baseline-Entscheidung.
- **Abnahme**: zwei Konfigurationen mit gleichem Token-Budget, ein
  Datenpunkt mehr Daten gegen einen kleineren Parameter-Count, Haupt-CE
  auf dem gemeinsamen Eval-Satz (`veylith/evaluate.py`).

### P2-3 Sparse MoE, aber nur mit belastbarer Umgebungsbasis

- **Problem und Evidenz**: Sparse ist numerisch gleichwertig, 12 bis 15 %
  schneller pro Step, 41 % sparsamer und laeuft dort, wo dense OOM hat
  (Batch 128, Block 512). Gleichzeitig faultet E16/K4 auf
  2.16.0.dev20261006, und das System-PyTorch ist 2.14.1.
- **Nutzen**: Faktor 5 weniger Rechenarbeit pro Token im Idealfall.
- **Aufwand**: M, abhaengig von der Umgebung.
- **Risiken**: Der Fault ist nicht behoben, nur layoutabhaengig
  verschoben. Keine Produktion, solange das nicht belegt ist. Expertenbreite
  als Vielfaches von 64 waere ein Kandidat, ist aber ungetestet.
- **Abnahme**: Fault-freie Laeufe ueber mindestens 10 000 Iterationen mit
  der Zielkonfiguration, dokumentierte PyTorch- und ROCm-Version, sonst
  keine Adoption.

### P2-4 Aux-loss-freie Balance

- **Problem und Evidenz**: Die Referenz-Configs nutzen `noaux_tc`, v2
  nutzt Switch-Style. Aktuell ist die Balance mit 1.018 pro Schicht
  bereits gut, der Aux-Term kostet aber 0.05 Loss und ein
  Gradientensignal am Router.
- **Nutzen**: moeglicherweise bessere Router-Qualitaet ohne Aux-Druck.
- **Aufwand**: M, **GPU-Freigabe noetig**.
- **Risiken**: Bias-Update ist eine neue Fehlerquelle (Kollaps).
- **Abnahme**: gleiche Balance (Streuung wie heute), Haupt-CE nicht
  schlechter, dokumentierte Kollaps-Pruefung.

### P2-5 Attention-Performance

- **Problem und Evidenz**: Die Maske wird pro Iteration neu gebaut, GQA
  wird per `repeat_interleave` materialisiert.
- **Nutzen**: Speicher und Rechenzeit, besonders bei groesserem Block.
- **Aufwand**: M.
- **Risiken**: SDPA aendert die Kernel-Pfade und damit potenziell die
  Stabilitaet auf dieser GPU. Nur nach P2-3-Klaerung und mit Messung.
- **Abnahme**: mindestens 10 % Step-Zeit bei identischer Loss-Kurve, kein
  neuer Fault.

## Empfohlener erster Schritt

**P0-1**, weil es der einzige Fehler ist, der in dieser Sitzung
tatsaechlich reproduziert wurde und direkt einen kompletten Lauf
zerstoert. Ausserdem ist er vollstaendig CPU-verifizierbar, aendert keine
Modellsemantik und keine Metrik, und er ist Voraussetzung fuer jedes
laengere Experiment danach.

Reihenfolge danach: P0-2 (Messwerte), P0-4 (Grenzfaelle), P0-3 (Doku),
dann P1-1 mit GPU-Freigabe.

## Nicht-Ziele dieses Auftrags

- Kein eigener Router-Service und kein neues Repository.
- Keine Wiederaufnahme der GPU-Forensik und keine neuen Fault-Reproducer.
- Kein Umbau von v1. v1 bleibt eingefroren.
- Keine Installation oder Aktualisierung von ROCm- oder Kernel-Komponenten.
- Keine Verteilung der 170 MB Binaerdaten aus der Git-Historie ohne
  ausdrueckliche Entscheidung.
