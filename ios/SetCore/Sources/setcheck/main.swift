// Testet den Erkennungskern der App am Mac. Ausgabe je Bild eine JSON-Zeile mit allen Erkennungen
// (Ecken, Wahrscheinlichkeiten) und dem Ergebnis der Gesamtkette (Karten, Sets).
// Querformat-Bilder werden wie in scripts/export_coreml.py im Uhrzeigersinn ins Hochformat gedreht.
import CoreImage
import CoreML
import Foundation
import SetCore

let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../../Models").standardized
let det = try MLModel.compileModel(at: models.appendingPathComponent("SetDetector.mlpackage"))
let cls = try MLModel.compileModel(at: models.appendingPathComponent("SetClassifier.mlpackage"))
let pipe = try LivePipeline(detectorURL: det, classifierURL: cls)

for path in CommandLine.arguments.dropFirst() {
    guard var img = CIImage(contentsOf: URL(fileURLWithPath: path), options: [.applyOrientationProperty: true]) else {
        FileHandle.standardError.write("kann \(path) nicht lesen\n".data(using: .utf8)!)
        continue
    }
    if img.extent.width > img.extent.height { img = img.oriented(.right) }
    img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
    // Rohe Erkennungen
    var dets: [[String: Any]] = []
    for d in try pipe.detector.detect(img) {
        let (p, pc) = try pipe.classifier.classify(img, corners: d.corners)
        dets.append(["c": d.corners.map { [$0.x, $0.y] }, "p": p, "pc": pc, "s": d.score])
    }
    // Gesamtkette: Standbild zweimal, damit das Tracking die Karten bestätigt (wie scripts/live.py)
    pipe.tracker.reset()
    var res = try pipe.process(img)
    let t = Date()
    res = try pipe.process(img)
    let ms = Date().timeIntervalSince(t) * 1000
    let line: [String: Any] = [
        "file": (path as NSString).lastPathComponent, "size": [img.extent.width, img.extent.height], "dets": dets,
        "cards": res.cards.map { $0.text }, "sets": res.sets.map { $0.map { res.cards[$0].text } },
        "ms": ms, "det_ms": res.detMs, "cls_ms": res.clsMs,
    ]
    print(String(data: try JSONSerialization.data(withJSONObject: line), encoding: .utf8)!)
}
