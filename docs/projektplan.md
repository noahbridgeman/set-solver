# SET Solver – Projektplan

Stand: 2026-11-15 (Vektor-Nachbau ergänzt) · Original-Doc: https://claude.ai/artifact/QjLcD2zTS8kkF8ezA8qXcT (diese Kopie wird nicht automatisch synchronisiert)

## Ziel und Anforderungen

Eine App erkennt Sets im Kartenspiel SET live aus dem Kamerabild und zeigt sie an.

- **Robustheit:** maximal, ML überall dort, wo es sinnvoll und nötig ist
- **Plattform:** Handy und Raspberry Pi
- **Kamera:** freihändig, unter anderem von oben
- **Laufzeit:** Echtzeit
- **Deck:** nur das Original-SET (81 Karten)
- **Zielgenauigkeit:** ≥ 99,9 % pro Attribut auf echten Fotos. Ein Set braucht 12 richtige Vorhersagen (3 Karten × 4 Attribute): bei 99 % pro Attribut sind nur ≈ 89 % der Sets richtig, bei 99,9 % ≈ 99 %.

## Architektur

Zweistufig: Ein Detektor findet die Karten, ein Klassifikator bestimmt die Attribute. Das ist datenschonender als ein Modell mit 81 Klassen und liefert pro Attribut eine Konfidenz.

1. **Karten finden (ML):** YOLO-pose mit 1 Klasse und 4 Eck-Keypoints pro Karte, Input 320–416 px.
2. **Entzerren:** Homographie aus den Ecken. Den Ausschnitt aus dem Frame in voller Auflösung nehmen, sonst sind die Streifen weg.
3. **Attribute erkennen (ML):** CNN (MobileNetV3-small) mit 4 Köpfen à 3 Klassen, Input ca. 128×192 px.
4. **Tracking:** ByteTrack oder ein IoU-Tracker. Die Wahrscheinlichkeiten pro Track werden über die Frames summiert.
5. **Solver:** Zu jedem Kartenpaar ist die dritte Karte bestimmt. Ein Set wird nur gemeldet, wenn die Konfidenz über einer Schwelle liegt.
6. **Ausgabe:** Handy: Overlay im Livebild. Pi: Display, später LEDs oder Laserpointer.

Der Klassifikator muss nicht jeden Frame laufen, z. B. nur alle 3–5 Frames oder bei neuen Tracks. Klassische Verfahren (Kontursuche, HSV) dienen nur noch zum Vorlabeln von Daten.

## Daten

Es gibt keine offiziellen Kartenbilder zum Download, deshalb wurde das eigene Deck fotografiert und als Vektormodell nachgebaut. Entscheidend ist allein die Genauigkeit auf echten Fotos.

**Kartenfotos (erledigt)**

- `data/raw/`: 9 Fotos mit je 3×3 Karten (Dateiname = `farbe_form`), dazu Einzelaufnahmen und ein Foto mit allen 3 Farben
- Labeln über die Sortierung: Zeile = Füllung (voll/gestreift/leer), Spalte = Anzahl (1/2/3)
- Warmes Kunstlicht; der Weißabgleich wird pro Karte über das Kartenweiß korrigiert

**Vektor-Nachbau (erledigt)**

Pipeline, alle Schritte neu erzeugbar (Laufzeit ≈ 3 min):

1. `scripts/crop_cards.py`: Karten finden, Kanten subpixelgenau fitten, auf 60 × 92 mm entzerren, Weißabgleich, automatisch labeln → `data/crops/`
2. `scripts/extract_geometry.py`: Symbolformen, Strich, Streifen, Layout und Farben messen → `data/geometry.json`
3. `scripts/render_cards.py`: alle 81 Karten als SVG und PNG in beliebiger Auflösung → `data/templates/`
4. `scripts/compare.py`: Render neben Foto für alle 81 Karten, mit Kennzahlen → `data/compare/`

Ergebnis: Konturabstand Render ↔ Foto im Median 0,09 mm, Form-IoU 0,985. Die Ausreißer kommen von gewölbten bzw. glänzenden Karten auf den Fotos, nicht vom Modell.

Erkenntnisse, die der Generator berücksichtigen muss:

- Die Welle ist nicht symmetrisch (auch nicht punktsymmetrisch). Eine um 180° gedrehte Karte sieht also anders aus, deshalb Drehungen um 180° in der Augmentierung mit abdecken.
- Konturen und Streifen sind dunkler gedruckt als die Flächen (Lila ΔE 21, Grün 13, Rot 6). Leere und gestreifte Symbole sind außen 0,27 bzw. 0,13 mm größer als volle.
- Der Symbolabstand hängt von der Form ab (Welle ≈ 23,5–23,8 mm, Raute/Oval ≈ 25,7–26,2 mm).
- Der Druck sitzt pro Karte um ≈ 0,8 mm versetzt (Schnitttoleranz), das sollte der Generator zufällig variieren. Gleiches gilt für leichte Wölbung der Karten.
- Streifen: Abstand 1,11 mm, Breite 0,31 mm. Für sicheres Erkennen braucht der Kartenausschnitt mindestens ≈ 3 px/mm, besser mehr.

**Synthetischer Generator (v1 steht)**

- `scripts/gen_synth.py` → `data/synth/v1/`: YOLO-pose-Daten (`detect/`, 4 Ecken, `flip_idx [1,0,3,2]`) und entzerrte Karten 128×192 mit 4 Attributen (`classify/`). Etwa 0,4 s pro Szene und Kern.
- Eck-Reihenfolge: im Bild im Uhrzeigersinn, k0–k1 = die obere kurze Kante. Entzerren k0..k3 → Hochformat.
- `scripts/preview_synth.py` zeigt die Labels zur Kontrolle. `scripts/export_real_crops.py` legt die 81 echten Karten als Validierungsset ab (mit und ohne Weißabgleich), nicht zum Training.
- Hintergründe: prozedural plus Ausschnitte aus den Rohfotos. Eigene Fotos von Tischen ohne Karten in `data/backgrounds/` erhöhen die Vielfalt.
- Grundlage ist der Renderer (`setcards/render.py`), nicht ausgeschnittene Fotos. Dadurch sind Auflösung, Druckversatz, Farben und Strichstärke frei variierbar.
- Karten auf zufällige Hintergründe setzen, mit Perspektive, Überlappung, Schatten, Bewegungs- und Fokusunschärfe, Rauschen, JPEG-Artefakten, Helligkeit und Weißabgleich
- Die Labels für Detektor und Klassifikator fallen dabei automatisch ab
- Den Farbton nur schwach verschieben, sonst lernt das Netz Rot = Lila. Weißabgleich und Farbtemperatur dagegen stark variieren.
- Die echten Crops aus `data/crops/` zusätzlich als Trainingsmaterial nutzen, damit das Netz nicht nur „perfekte“ Karten kennt

**Echte Daten**

- Einige hundert Frames aus Handyvideos, in verschiedenen Räumen und bei verschiedenem Licht, auch mit Blitz bzw. Taschenlampe
- Dienen als Testset und zum Nachtrainieren, halbautomatisch gelabelt: das Modell schlägt vor, du korrigierst von Hand
- Fremde Datensätze als Zusatz und als zweites Testset: Roboflow „Set“ (https://universe.roboflow.com/venividireliqui/set-vxd4a, nach den 4 Attributen gelabelt) und Roboflow „SETCARDSBATCH“ (https://universe.roboflow.com/tel-aviv/setcardsbatch/dataset/3, 482 Bilder). Lizenz und Labelqualität prüfen.

## Plattform und Hardware

Zuerst das Handy als Ziel, dort sind Kamera und Rechenleistung besser. Der Pi folgt mit denselben Modellen.

| Ziel | Laufzeit | Hinweis |
| --- | --- | --- |
| Training | PyTorch / Ultralytics am PC | Export für beide Ziele aus einem Modell |
| Handy | TFLite (Android) oder CoreML (iOS) auf dem NPU | realistisch flüssig |
| Raspberry Pi 5 | nur CPU: eher einstellige Bildrate; mit AI HAT+ (Hailo): Echtzeit | Camera Module 3 mit Autofokus, Fokus ist für die Streifen relevant |

## Phasenplan

Nächster Schritt ist die Echtzeit-Pipeline am PC; parallel das echte Testset (1c). Phase 6 läuft als Schleife, bis die Zielgenauigkeit erreicht ist.

- [x] 1a. 81 Karten fotografieren, ausschneiden und labeln
- [x] 1b. Vektor-Nachbau aller 81 Karten (Renderer + Vergleich mit den Fotos)
- [ ] 1c. Echtes Testset aufnehmen (Handyvideos, Labeln später)
- [x] 2. Synthetischen Datengenerator auf Basis des Renderers bauen (v1; wird anhand der echten Fehler weiter verbessert)
- [x] 3. Detektor mit Eck-Keypoints trainieren (v1, siehe Ergebnisse)
- [x] 4. Attribut-Klassifikator trainieren (v1, siehe Ergebnisse)
- [ ] 5. Echtzeit-Pipeline am PC: Webcam oder Handy-Stream, dazu Tracking, Solver und Overlay
- [ ] 6. Auf echten Daten auswerten, Fehler analysieren, nachtrainieren, wiederholen
- [ ] 7. Auf Handy portieren, danach auf den Pi

## Ergebnisse v1 (2026-11-16)

Alle Modelle nur auf synthetischen Daten trainiert; die echten Fotos dienen nur zur Kontrolle.

| Modell | Synthetische Validierung | Echte Fotos (17 Fotos, 91 Karten) |
| --- | --- | --- |
| Klassifikator `models/classifier_v1.pt` (MobileNetV3-small, 15 Epochen, Bestwert Epoche 11) | 99,96 % ganze Karte (6 von 13.642 falsch, alle lila Umrisse bei extremer Unschärfe oder Farbstich) | 100 %, mit und ohne Weißabgleich |
| Detektor `models/detector_v1.pt` (YOLO26n-pose, 640 px, 20 Epochen, ca. 16 min/Epoche, Batch 16) | Box mAP50-95 0,982, Ecken mAP50-95 0,995 | alle 91 Karten gefunden; Ecken mAP50-95 0,995 |
| Gesamtkette `scripts/eval_real_pipeline.py` (Detektor → Entzerren aus dem Originalfoto → Klassifikator mit 0°/180°) | – | 91/91 Karten komplett richtig, 0 Fehlalarme bei conf 0,6 |

Erkenntnisse:

- Eckfehler des Detektors auf echten Fotos: Median 2,8 %, maximal 5,1 % der kurzen Kartenkante (ca. 1,7 bzw. 3 mm), ohne systematische Verschiebung. Das reicht für den Klassifikator; für eine genauere Lage kann man die Ecken im Originalbild per Kantenfit nachschärfen (wie `refine_corners_fullres`).
- Bei conf 0,5 gab es einen Fehlalarm (weißes Tuch am Bildrand, conf 0,50) → Schwelle 0,6.
- Training auf dem Mac: Batch 32 mit 6 Workern sprengt die 16 GB RAM. Funktioniert mit `--batch 16 --workers 3` und `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`.
- Das echte Testset ist klein und einfach (Draufsicht, dunkler Grund, wenige Karten pro Foto). Aussagekräftig wird erst das Video-Testset (1c).

## Offene Fragen

- iOS oder Android? Das entscheidet zwischen CoreML/Swift, TFLite/Kotlin oder Flutter für beides.
- Lizenz der Roboflow-Datensätze klären.

Vergleichsmaßstab: Die iOS-App SET Detector (https://klukacsy.github.io/SET-detector.github.io) arbeitet ebenfalls zweistufig: erst Karten finden, dann jede Karte einzeln klassifizieren.
