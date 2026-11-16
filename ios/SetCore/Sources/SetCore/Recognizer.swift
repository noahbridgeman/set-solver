// Erkennung mit den CoreML-Modellen aus scripts/export_coreml.py. Gleiche Schritte wie setcards/live.py:
// Detektor (letterboxed 384x640) -> Ecken zurückrechnen -> Karte entzerren (128x192) -> Klassifikator.
// Alle Koordinaten in Bildpixeln mit Ursprung oben links (wie OpenCV); CoreImage rechnet intern von unten.
import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation

public struct Detection: Sendable {
    public var corners: [CGPoint]
    public var score: Float
}

public enum RecognizerError: Error { case pixelBuffer, output }

func makeBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                                  kCVPixelBufferCGImageCompatibilityKey: true,
                                  kCVPixelBufferCGBitmapContextCompatibilityKey: true]
    guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            == kCVReturnSuccess, let b = pb else { throw RecognizerError.pixelBuffer }
    return b
}

/// CIContext ohne Farbmanagement: Pixelwerte gehen unverändert durch, wie bei OpenCV in Python.
public func makeRawContext() -> CIContext {
    CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false])
}

public final class CardDetector {
    public static let inW = 384, inH = 640
    let model: MLModel
    let ctx: CIContext
    let buffer: CVPixelBuffer

    public init(modelURL: URL, context: CIContext) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: cfg)
        ctx = context
        buffer = try makeBuffer(Self.inW, Self.inH)
    }

    /// image: Bild in CoreImage-Koordinaten, extent (0, 0, W, H).
    public func detect(_ image: CIImage, conf: Float = 0.6, iou: Float = 0.7) throws -> [Detection] {
        let W = image.extent.width, H = image.extent.height
        let r = min(CGFloat(Self.inH) / H, CGFloat(Self.inW) / W)
        let nw = (W * r).rounded(), nh = (H * r).rounded()
        let left = floor((CGFloat(Self.inW) - nw) / 2), top = floor((CGFloat(Self.inH) - nh) / 2)
        let rect = CGRect(x: 0, y: 0, width: Self.inW, height: Self.inH)
        let gray = CIImage(color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255)).cropped(to: rect)
        let placed = image
            .transformed(by: CGAffineTransform(scaleX: nw / W, y: nh / H))
            .transformed(by: CGAffineTransform(translationX: left, y: CGFloat(Self.inH) - top - nh))
        ctx.render(placed.composited(over: gray), to: buffer, bounds: rect, colorSpace: nil)

        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]))
        guard let name = model.modelDescription.outputDescriptionsByName.keys.first,
              let arr = out.featureValue(for: name)?.multiArrayValue else { throw RecognizerError.output }
        // Form (1, 17, N): Mitte x, Mitte y, Breite, Höhe, Sicherheit, 4 x (x, y, sichtbar)
        let s1 = arr.strides[1].intValue, s2 = arr.strides[2].intValue, n = arr.shape[2].intValue
        var cand: [(box: CGRect, score: Float, corners: [CGPoint])] = []
        func read(_ p: UnsafeBufferPointer<Float>) {
            for a in 0..<n where p[4 * s1 + a * s2] >= conf {
                let v = { (row: Int) in CGFloat(p[row * s1 + a * s2]) }
                let box = CGRect(x: v(0) - v(2) / 2, y: v(1) - v(3) / 2, width: v(2), height: v(3))
                let corners = (0..<4).map { k in
                    CGPoint(x: min(max((v(5 + 3 * k) - left) / r, 0), W),
                            y: min(max((v(6 + 3 * k) - top) / r, 0), H))
                }
                cand.append((box, p[4 * s1 + a * s2], corners))
            }
        }
        if arr.dataType == .float32 {
            arr.withUnsafeBufferPointer(ofType: Float.self) { read($0) }
        } else {  // falls CoreML Float16 liefert
            var f = [Float](repeating: 0, count: arr.count)
            for i in 0..<arr.count { f[i] = arr[i].floatValue }
            f.withUnsafeBufferPointer { read($0) }
        }
        // NMS wie Ultralytics (eine Klasse): höchste Sicherheit zuerst, stark Überlappende verwerfen
        cand.sort { $0.score > $1.score }
        var keep: [Int] = []
        for i in cand.indices {
            let b = cand[i].box
            let ok = keep.allSatisfy { j in
                let inter = b.intersection(cand[j].box)
                let ia = inter.isNull ? 0 : inter.width * inter.height
                return Float(ia / (b.width * b.height + cand[j].box.width * cand[j].box.height - ia + 1e-9)) <= iou
            }
            if ok { keep.append(i) }
        }
        return keep.map { Detection(corners: cand[$0].corners, score: cand[$0].score) }
    }
}

public final class CardClassifier {
    public static let inW = 128, inH = 192
    let model: MLModel
    let ctx: CIContext
    let buffer: CVPixelBuffer

    public init(modelURL: URL, context: CIContext) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: cfg)
        ctx = context
        buffer = try makeBuffer(Self.inW, Self.inH)
    }

    /// Karte entzerren: Ecke 0 -> oben links, 1 -> oben rechts, 2 -> unten rechts, 3 -> unten links.
    public func warp(_ image: CIImage, corners k: [CGPoint]) -> CIImage {
        let H = image.extent.height
        let ci = k.map { CIVector(x: $0.x, y: H - $0.y) }  // in CoreImage-Koordinaten (y nach oben)
        let f = CIFilter(name: "CIPerspectiveCorrection")!
        f.setValue(image.clampedToExtent(), forKey: kCIInputImageKey)  // Rand fortsetzen wie BORDER_REPLICATE
        f.setValue(ci[0], forKey: "inputTopLeft")
        f.setValue(ci[1], forKey: "inputTopRight")
        f.setValue(ci[2], forKey: "inputBottomRight")
        f.setValue(ci[3], forKey: "inputBottomLeft")
        var out = f.outputImage!
        out = out.transformed(by: CGAffineTransform(translationX: -out.extent.minX, y: -out.extent.minY))
        // auf 128x192 bringen: Lanczos glättet beim Verkleinern (wie INTER_AREA in Python)
        let sy = CGFloat(Self.inH) / out.extent.height, sx = CGFloat(Self.inW) / out.extent.width
        let l = CIFilter(name: "CILanczosScaleTransform")!
        l.setValue(out, forKey: kCIInputImageKey)
        l.setValue(sy, forKey: kCIInputScaleKey)
        l.setValue(sx / sy, forKey: kCIInputAspectRatioKey)
        out = l.outputImage!
        out = out.transformed(by: CGAffineTransform(translationX: -out.extent.minX, y: -out.extent.minY))
        return out.cropped(to: CGRect(x: 0, y: 0, width: Self.inW, height: Self.inH))
    }

    /// -> 4 Attribute x 3 Wahrscheinlichkeiten und P(ist Karte).
    public func classify(_ image: CIImage, corners: [CGPoint]) throws -> (probs: [[Float]], pCard: Float) {
        let rect = CGRect(x: 0, y: 0, width: Self.inW, height: Self.inH)
        ctx.render(warp(image, corners: corners), to: buffer, bounds: rect, colorSpace: nil)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]))
        guard let arr = out.featureValue(for: "probs")?.multiArrayValue else { throw RecognizerError.output }
        let p = (0..<14).map { arr[$0].floatValue }
        return ((0..<4).map { Array(p[3 * $0..<3 * $0 + 3]) }, p[13])
    }
}

public struct FrameResult: Sendable {
    public var cards: [CardInfo]
    public var sets: [[Int]]             // Indizes in cards
    public var detMs: Double = 0
    public var clsMs: Double = 0
    public var imageSize: CGSize = .zero
}

/// Gesamtkette wie LivePipeline in setcards/live.py.
public final class LivePipeline {
    public let detector: CardDetector
    public let classifier: CardClassifier
    public let tracker = Tracker()
    public var minConf: Float = 0.8
    public var maxMissesShow = 2
    public var edgeMargin: CGFloat = 0.015
    public var edgeMinCard: Float = 0.9
    public var edgeMinConf: Float = 0.9

    public init(detectorURL: URL, classifierURL: URL) throws {
        let ctx = makeRawContext()
        detector = try CardDetector(modelURL: detectorURL, context: ctx)
        classifier = try CardClassifier(modelURL: classifierURL, context: ctx)
    }

    public func process(_ image: CIImage) throws -> FrameResult {
        let t0 = Date()
        let dets = try detector.detect(image)
        let t1 = Date()
        var corners: [[CGPoint]] = [], probs: [[[Float]]] = [], valid: [Bool] = []
        let W = image.extent.width, H = image.extent.height, m = edgeMargin * min(W, H)
        for d in dets {
            let (p, pCard) = try classifier.classify(image, corners: d.corners)
            if pCard < 0.5 { continue }  // Fehlalarme des Detektors (Untergrund, Papier, Tuch) verwerfen
            // Karten am Rand zählen nur, wenn sie ganz drin sind oder der Klassifikator sich sicher ist
            let inside = d.corners.allSatisfy { $0.x >= m && $0.x <= W - 1 - m && $0.y >= m && $0.y <= H - 1 - m }
            let sure = pCard >= edgeMinCard && p.map { $0.max()! }.min()! >= edgeMinConf
            corners.append(d.corners)
            probs.append(p)
            valid.append(inside || sure)
        }
        let t2 = Date()
        let tracks = tracker.update(corners: corners, probs: probs, valid: valid)
        let cards = tracks.filter { $0.confirmed && $0.misses <= maxMissesShow }.map(CardInfo.init)
        // Sets nur aus Karten, die gerade sicher erkannt sind
        let sure = cards.indices.filter { cards[$0].confidence >= minConf }
        let sets = SetSolver.findSets(sure.map { cards[$0].label }).map { $0.map { sure[$0] } }
        return FrameResult(cards: cards, sets: sets, detMs: t1.timeIntervalSince(t0) * 1000,
                           clsMs: t2.timeIntervalSince(t1) * 1000, imageSize: CGSize(width: W, height: H))
    }
}
