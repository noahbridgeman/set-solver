// SET-Regel (wie setcards/solver.py): drei Karten bilden ein Set, wenn jedes Attribut bei allen gleich
// oder bei allen verschieden ist. Karten als 4 Klassenindizes 0...2; gleich oder alle verschieden <=> Summe % 3 == 0.
import Foundation

public enum SetSolver {
    public static func isSet(_ a: [Int], _ b: [Int], _ c: [Int]) -> Bool {
        zip(zip(a, b), c).allSatisfy { ($0.0 + $0.1 + $1) % 3 == 0 }
    }

    /// Die eine Karte, die a und b zu einem Set ergänzt.
    public static func thirdCard(_ a: [Int], _ b: [Int]) -> [Int] {
        zip(a, b).map { (6 - $0 - $1) % 3 }
    }

    /// Alle Sets als Indextripel (i < j < k), sortiert.
    public static func findSets(_ cards: [[Int]]) -> [[Int]] {
        var pos: [[Int]: [Int]] = [:]
        for (i, c) in cards.enumerated() { pos[c, default: []].append(i) }
        var out = Set<[Int]>()
        for i in 0..<cards.count {
            for j in (i + 1)..<max(i + 1, cards.count) {
                for k in pos[thirdCard(cards[i], cards[j])] ?? [] where k != i && k != j {
                    out.insert([i, j, k].sorted())
                }
            }
        }
        return out.sorted { $0.lexicographicallyPrecedes($1) }
    }
}
