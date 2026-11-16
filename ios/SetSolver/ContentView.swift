// Anzeige: Kamerabild mit Kartenrahmen, Beschriftungen und Sets (wie draw/hud in setcards/live.py).
// Tippen aufs Bild: nächstes Set einzeln zeigen; Knopf „Alle“: wieder alle Sets.
import SetCore
import SwiftUI

private let setColors: [Color] = [
    Color(red: 1, green: 0, blue: 1), Color(red: 0, green: 1, blue: 1), Color(red: 1, green: 0.65, blue: 0),
    Color(red: 0, green: 1, blue: 0), Color(red: 0, green: 0.4, blue: 1), Color(red: 1, green: 1, blue: 0),
    Color(red: 1, green: 0.41, blue: 0.71), Color(red: 1, green: 0, blue: 0.5),
]

struct ContentView: View {
    @StateObject private var cam = CameraModel()
    @State private var showLabels = true
    @State private var onlySet: Int?
    private let minConf: Float = 0.8

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let frame = cam.frame {
                GeometryReader { geo in
                    let iw = CGFloat(frame.width), ih = CGFloat(frame.height)
                    let s = min(geo.size.width / iw, geo.size.height / ih)
                    let rect = CGRect(x: (geo.size.width - iw * s) / 2, y: (geo.size.height - ih * s) / 2,
                                      width: iw * s, height: ih * s)
                    ZStack(alignment: .topLeading) {
                        Image(decorative: frame, scale: 1)
                            .resizable()
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                        Canvas { gc, _ in
                            if let res = cam.result { draw(res, in: &gc, scale: s, origin: rect.origin) }
                        }
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { nextSet() }
            } else {
                Text(cam.message ?? "Kamera startet …")
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding()
            }
            VStack {
                hud
                Spacer()
                controls
            }
            .padding(.horizontal, 12)
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            cam.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            cam.stop()
        }
    }

    // MARK: - Oben: Anzahl Karten/Sets

    private var hud: some View {
        let n = cam.result?.cards.count ?? 0
        let sets = cam.result?.sets.count ?? 0
        var line = "\(n) Karten   " + (sets == 0 ? "kein Set" : "\(sets) Set\(sets == 1 ? "" : "s")")
        if let o = onlySet, sets > 0 { line += "   (\(o % sets + 1)/\(sets))" }
        let ms = (cam.result?.detMs ?? 0) + (cam.result?.clsMs ?? 0)
        return VStack(alignment: .leading, spacing: 2) {
            Text(line).font(.headline)
            Text(String(format: "%.0f fps · %.0f ms", cam.fps, ms)).font(.caption2).opacity(0.8)
            if let m = cam.message { Text(m).font(.caption) }
        }
        .foregroundStyle(.white)
        .padding(8)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Unten: Knöpfe

    private var controls: some View {
        HStack(spacing: 10) {
            if cam.hasUltraWide {
                button(cam.ultraWide ? "0.5×" : "1×") { cam.setUltraWide(!cam.ultraWide) }
            }
            button(onlySet == nil ? "Alle Sets" : "Einzeln") {
                onlySet = onlySet == nil ? 0 : nil
            }
            button(showLabels ? "Text an" : "Text aus") { showLabels.toggle() }
            button(cam.paused ? "Weiter" : "Pause") { cam.setPaused(!cam.paused) }
        }
        .padding(.bottom, 8)
    }

    private func button(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.bold())
                .frame(minWidth: 64)
                .padding(.vertical, 10)
                .padding(.horizontal, 8)
                .background(.black.opacity(0.6), in: Capsule())
                .foregroundStyle(.white)
        }
    }

    private func nextSet() {
        guard let n = cam.result?.sets.count, n > 0 else { return }
        onlySet = (onlySet ?? -1) + 1
    }

    // MARK: - Zeichnen

    private func draw(_ res: FrameResult, in gc: inout GraphicsContext, scale s: CGFloat, origin o: CGPoint) {
        func pt(_ p: CGPoint) -> CGPoint { CGPoint(x: o.x + p.x * s, y: o.y + p.y * s) }
        func quad(_ k: [CGPoint], grow f: CGFloat = 1) -> Path {
            let c = CGPoint(x: k.map(\.x).reduce(0, +) / 4, y: k.map(\.y).reduce(0, +) / 4)
            let g = k.map { CGPoint(x: c.x + ($0.x - c.x) * f, y: c.y + ($0.y - c.y) * f) }
            var p = Path()
            p.addLines(g.map(pt))
            p.closeSubpath()
            return p
        }
        for c in res.cards {
            let sure = c.confidence >= minConf
            gc.stroke(quad(c.corners), with: .color(sure ? .white.opacity(0.8) : .yellow), lineWidth: sure ? 1.5 : 2.5)
        }
        var sets = Array(res.sets.enumerated())
        if let only = onlySet, !sets.isEmpty { sets = [sets[only % sets.count]] }
        var layer: [Int: Int] = [:]
        for (si, set) in sets {
            for i in set {
                let n = layer[i, default: 0]
                layer[i] = n + 1
                gc.stroke(quad(res.cards[i].corners, grow: 1.04 + 0.05 * CGFloat(n)),
                          with: .color(setColors[si % setColors.count]), lineWidth: 4)
            }
        }
        guard showLabels else { return }
        for c in res.cards {
            let sure = c.confidence >= minConf
            let txt = c.text + (sure ? "" : String(format: " ?%.2f", c.confidence))
            let t = gc.resolve(Text(txt).font(.caption.bold()).foregroundColor(.white))
            let size = t.measure(in: CGSize(width: 400, height: 50))
            let m = pt(CGPoint(x: c.corners.map(\.x).reduce(0, +) / 4, y: c.corners.map(\.y).reduce(0, +) / 4))
            let r = CGRect(x: m.x - size.width / 2 - 4, y: m.y - size.height / 2 - 2,
                           width: size.width + 8, height: size.height + 4)
            gc.fill(Path(roundedRect: r, cornerRadius: 4), with: .color(.black.opacity(0.75)))
            gc.draw(t, at: m)
        }
    }
}
