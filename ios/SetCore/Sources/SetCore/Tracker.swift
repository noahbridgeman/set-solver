// Tracking wie in setcards/live.py: Erkennungen werden von Bild zu Bild über die Kartenmitte zugeordnet
// (Hungarian). Ecken und Wahrscheinlichkeiten werden geglättet, damit Labels nicht flackern.
import CoreGraphics
import Foundation

public let attributeNames: [[String]] = [
    ["rot", "grün", "lila"],
    ["Raute", "Oval", "Welle"],
    ["voll", "gestreift", "leer"],
    ["1", "2", "3"],
]

public final class Track {
    public let id: Int
    public var corners: [CGPoint]        // 4 Ecken, Ecke 0-1 = obere kurze Kante, Bildpixel (y nach unten)
    public var probs: [[Float]]          // 4 Attribute x 3 Klassen
    public var hits = 1
    public var misses = 0
    public var nGood = 0                 // Bilder mit sicherer Einzelerkennung
    public var confirmed = false

    init(id: Int, corners: [CGPoint], probs: [[Float]]) {
        self.id = id
        self.corners = corners
        self.probs = probs
    }

    public var label: [Int] { probs.map { p in p.indices.max { p[$0] < p[$1] }! } }
    public var confidence: Float { probs.map { $0.max()! }.min()! }
    public var text: String { zip(attributeNames, label).map { $0[$1] }.joined(separator: " ") }
    public var center: CGPoint { mean(corners) }
}

/// Unveränderliche Kopie einer Karte für die Anzeige (Tracks ändern sich im Erkennungs-Thread weiter).
public struct CardInfo: Sendable {
    public let id: Int
    public let corners: [CGPoint]
    public let label: [Int]
    public let confidence: Float
    public let text: String

    init(_ t: Track) {
        id = t.id
        corners = t.corners
        label = t.label
        confidence = t.confidence
        text = t.text
    }
}

func mean(_ p: [CGPoint]) -> CGPoint {
    CGPoint(x: p.map(\.x).reduce(0, +) / CGFloat(p.count), y: p.map(\.y).reduce(0, +) / CGFloat(p.count))
}

func dist(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(a.x - b.x, a.y - b.y) }

public final class Tracker {
    public var gate: CGFloat = 0.6
    public var maxMisses = 4
    public var kpAlpha: CGFloat = 0.7
    public var probAlpha: Float = 0.35
    public var minConf: Float = 0.8
    public var confirmFrames = 2
    public var dropConf: Float = 0.5
    public private(set) var tracks: [Track] = []
    private var nextId = 0

    public init() {}

    public func reset() { tracks.removeAll() }

    private func observe(_ t: Track, _ p: [[Float]], _ valid: Bool) {
        if valid, p.map({ $0.max()! }).min()! >= minConf { t.nGood += 1 }
        if t.nGood >= confirmFrames && t.confidence >= minConf {
            t.confirmed = true
        } else if t.confidence < dropConf {
            t.confirmed = false
        }
    }

    /// valid[j] = false: Karte ragt aus dem Bild -> Lage nachführen, Labels nicht anfassen.
    public func update(corners: [[CGPoint]], probs: [[[Float]]], valid: [Bool]) -> [Track] {
        let T = tracks.count, D = corners.count
        var matchedT = Set<Int>(), matchedD = Set<Int>()
        if T > 0 && D > 0 {
            let dc = corners.map(mean)
            var cost = [[Double]](repeating: [Double](repeating: 0, count: D), count: T)
            for (i, t) in tracks.enumerated() {
                let tc = t.center
                let scale = max(shortEdge(t.corners), 1)
                for j in 0..<D { cost[i][j] = Double(dist(tc, dc[j]) / scale) }
            }
            for (i, j) in hungarian(cost) where cost[i][j] < Double(gate) {
                let t = tracks[i]
                // Reihenfolge kann um 180° springen -> an alte Ecken angleichen
                let k = [corners[j], roll2(corners[j])].min { a, b in sqDiff(a, t.corners) < sqDiff(b, t.corners) }!
                t.corners = zip(k, t.corners).map { n, o in
                    CGPoint(x: kpAlpha * n.x + (1 - kpAlpha) * o.x, y: kpAlpha * n.y + (1 - kpAlpha) * o.y)
                }
                if valid[j] {
                    t.probs = zip(probs[j], t.probs).map { n, o in zip(n, o).map { probAlpha * $0 + (1 - probAlpha) * $1 } }
                }
                observe(t, probs[j], valid[j])
                t.hits += 1
                t.misses = 0
                matchedT.insert(i)
                matchedD.insert(j)
            }
        }
        for (i, t) in tracks.enumerated() where !matchedT.contains(i) { t.misses += 1 }
        tracks.removeAll { $0.misses > maxMisses }
        for j in 0..<D where !matchedD.contains(j) {
            let t = Track(id: nextId, corners: corners[j], probs: probs[j])
            observe(t, probs[j], valid[j])
            tracks.append(t)
            nextId += 1
        }
        return tracks
    }
}

func shortEdge(_ k: [CGPoint]) -> CGFloat { min(dist(k[1], k[0]), dist(k[3], k[2])) }
func roll2(_ k: [CGPoint]) -> [CGPoint] { [k[2], k[3], k[0], k[1]] }
func sqDiff(_ a: [CGPoint], _ b: [CGPoint]) -> CGFloat {
    zip(a, b).map { pow($0.x - $1.x, 2) + pow($0.y - $1.y, 2) }.reduce(0, +)
}

/// Minimale Zuordnung (Hungarian / Kuhn-Munkres) für eine rechteckige Kostenmatrix.
/// Gibt Paare (Zeile, Spalte) zurück, höchstens min(Zeilen, Spalten) viele.
func hungarian(_ cost: [[Double]]) -> [(Int, Int)] {
    let rows = cost.count, cols = cost.first?.count ?? 0
    if rows == 0 || cols == 0 { return [] }
    let n = max(rows, cols)
    let big = (cost.flatMap { $0 }.max() ?? 0) * 10 + 1
    func c(_ i: Int, _ j: Int) -> Double { i < rows && j < cols ? cost[i][j] : big }
    // e-maxx-Variante, 1-basiert
    var u = [Double](repeating: 0, count: n + 1), v = u
    var p = [Int](repeating: 0, count: n + 1), way = p
    for i in 1...n {
        p[0] = i
        var j0 = 0
        var minv = [Double](repeating: .infinity, count: n + 1)
        var used = [Bool](repeating: false, count: n + 1)
        repeat {
            used[j0] = true
            let i0 = p[j0]
            var delta = Double.infinity, j1 = 0
            for j in 1...n where !used[j] {
                let cur = c(i0 - 1, j - 1) - u[i0] - v[j]
                if cur < minv[j] { minv[j] = cur; way[j] = j0 }
                if minv[j] < delta { delta = minv[j]; j1 = j }
            }
            for j in 0...n {
                if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
            }
            j0 = j1
        } while p[j0] != 0
        repeat {
            let j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        } while j0 != 0
    }
    var out: [(Int, Int)] = []
    for j in 1...n where p[j] - 1 < rows && j - 1 < cols { out.append((p[j] - 1, j - 1)) }
    return out.sorted { $0.0 < $1.0 }
}
