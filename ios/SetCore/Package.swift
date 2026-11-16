// swift-tools-version:5.10
// Nur für Tests am Mac: `swift run -c release setcheck <Bilder...>` läuft den Erkennungskern der App
// (SetCore) auf Fotos. Die iPhone-App selbst wird über ../SetSolver.xcodeproj gebaut.
import PackageDescription

let package = Package(
    name: "SetCore",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "SetCore", targets: ["SetCore"])],
    targets: [
        .target(name: "SetCore"),
        .executableTarget(name: "setcheck", dependencies: ["SetCore"]),
    ]
)
