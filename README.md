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

## 4. Reproduktion der Ergebnisse

Die numerische Analyse erfolgt in drei Schritten:

1. **Modelllösung:** Berechnung der stationären Lösung für die betrachteten Parameterkombinationen.
2. **Auswertung:** Bestimmung der Wahlwahrscheinlichkeiten, der zusätzlichen Informationsverarbeitung, der Bindungsprämie und der Anbieterergebnisse.
3. **Darstellung und Validierung:** Erstellung der Abbildungen und Tabellen sowie Durchführung der Robustheits- und Plausibilitätsprüfungen.

Die verwendete Baseline-Parametrisierung, die numerischen Einstellungen und die untersuchten Parameterbereiche sind in der Bachelorarbeit dokumentiert.

Da die vollständigen Berechnungen rechenintensiv sein können, werden Ergebnisse und Zwischenergebnisse gespeichert. Dadurch kann die Ausführung bei Bedarf unterbrochen und später fortgesetzt werden. Zudem lassen sich Abbildungen nachträglich anpassen.

## 5. Hinweise

Die Modellparameter sind bewusst stilisiert und stellen keine empirische Kalibrierung dar.

Die theoretischen Grundlagen, die mathematische Modellformulierung sowie die Interpretation der Ergebnisse sind in der Bachelorarbeit erläutert.
