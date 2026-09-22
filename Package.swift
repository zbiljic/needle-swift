// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "needle-swift",
    platforms: [.macOS(.v13)],
    products: [.library(name: "Needle", targets: ["Needle"])],
    targets: [
        .target(name: "Needle"),
        .testTarget(name: "NeedleTests", dependencies: ["Needle"]),
    ]
)
