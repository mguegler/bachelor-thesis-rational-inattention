# Packungsgrössenwahl unter Rational Inattention

**Informationskosten, Zustandspersistenz und der Wert der Entscheidungsflexibilität**

Dieses Repository enthält den Python-Code zur Bachelorarbeit *«Packungsgrössenwahl unter Rational Inattention»*.

## 1. Projektübersicht

Die Arbeit untersucht, wie kostspielige Informationsverarbeitung und die Persistenz zeitlich veränderlicher Qualitätsunterschiede die Produkt- und Packungsgrössenwahl sowie den gewinnmaximierenden Mengenrabatt beeinflussen.

Dazu wird ein dynamisches Rational-Inattention-Modell mit zwei Produkten in jeweils zwei Packungsgrössen numerisch gelöst. Grosspackungen bieten einen Mengenrabatt, binden die konsumierende Person jedoch für eine zusätzliche Periode an das gewählte Produkt.

## 2. Projektstruktur

Die zentralen Python-Dateien sind:
- **`modell_parameter.py`** – Definition und Validierung der Modellparameter sowie Festlegung der Parameterraster für die Abbildungen.
- **`modell_konsument.py`** – Numerische Lösung des dynamischen Rational-Inattention-Modells.
- **`modell_anbieter.py`** – Berechnung gewinnbezogener Kennzahlen.
- **`auswertung.py`** – Berechnung und Speicherung der Modellergebnisse sowie Erstellung der Abbildungen und Durchführung der Robustheitsprüfungen.

Die Jupyter-Notebooks dienen der Durchführung der Berechnungen und der Erstellung der in der Arbeit verwendeten Ergebnisse.

## 3. Voraussetzungen

Für die Ausführung wird Python benötigt. Die verwendeten Python-Bibliotheken und deren Versionen sind in `requirements.txt` dokumentiert.

Die Abhängigkeiten können mit folgendem Befehl installiert werden:
```bash
python -m pip install -r requirements.txt
```

## 4. Reproduktion der Ergebnisse

Die Notebooks befinden sich im selben Verzeichnis wie die Python-Dateien und werden in folgender Reihenfolge ausgeführt:
1. **`modellanalyse.ipynb`**
2. **`anhang.ipynb`**

Dadurch werden folgende Ordner erstellt:
- **`resultate/`** – Zwischengespeicherte Modellergebnisse und Solver-Diagnosen.
- **`abbildungen/`** – Abbildungen als PDF.
- **`tabellen_anhang/`** – Robustheitstabellen als CSV.

Die verwendete Baseline-Parametrisierung, die numerischen Einstellungen und die untersuchten Parameterbereiche sind in der Bachelorarbeit dokumentiert.

Da die vollständigen Berechnungen rechenintensiv sein können, werden die berechneten Ergebnisse zwischengespeichert. Dadurch kann die Ausführung bei Bedarf unterbrochen und später fortgesetzt werden. Zudem lassen sich Abbildungen nachträglich anpassen.

## 5. Hinweise

Die Modellparameter sind bewusst stilisiert und stellen keine empirische Kalibrierung dar.

Die theoretischen Grundlagen, die mathematische Modellformulierung sowie die Interpretation der Ergebnisse sind in der Bachelorarbeit erläutert.
