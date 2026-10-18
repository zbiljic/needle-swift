import Foundation

struct EngineArtifact: Sendable {
    let name: String
    let url: String
    let checksum: String
    var archivePath: String?
}

/// One downloadable release: the engine libraries per platform and, when the
/// engine needs them, the weights archive published beside them.
struct EngineRelease: Sendable {
    let generation: Int
    let version: String
    let libraryName: String
    private let libraries: [Engine.Platform: EngineArtifact]
    /// The archive loaded with `needle_load`; nil for Needle 2, whose engine embeds its model.
    let weights: EngineArtifact?

    static let needle2 = pin(
        generation: 2,
        version: "2.0.4",
        source: ("Cactus-Compute/needle2", "32e9e3a93b205f786929697446ae669cf0a84579"),
        libraryName: "libneedle.dylib",
        checksums: [
            .darwinARM64: "abae4cca0a4d84ec73da4bde18803b9be812a9209e48fb7fa372002ebaa60265",
            .darwinAMD64: "071e93d996021b4f6b5bee055777cd81be9f6963d2565115ec461b9f71c7245e",
        ]
    )

    static let needle3 = pin(
        generation: 3,
        version: Engine.version,
        source: ("Cactus-Compute/needle3", "c7c415a3d1b3d929014bc6e866d51ebb971f7089"),
        libraryName: "libneedle3.dylib",
        checksums: [
            .darwinARM64: "ad1cba80ede4c058692370964eec4d881a1fdc80014f7210b2d13891bad7d1c6",
            .darwinAMD64: "31a90adbaa898ea65e92b370e80ba5830a346348022a913a8bb254ab7d20bad6",
        ],
        weights: ("needle3.cact", "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38")
    )

    /// Speech weights run on the Needle 3 engine and share its cache directory.
    static let speechWeights = EngineArtifact(
        name: "whistle.cact",
        url: "https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact",
        checksum: "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"
    )

    static let whistle = Self(
        generation: 3,
        version: needle3.version,
        libraryName: needle3.libraryName,
        libraries: needle3.libraries,
        weights: speechWeights
    )

    private static func pin(
        generation: Int,
        version: String,
        source: (repository: String, revision: String),
        libraryName: String,
        checksums: [Engine.Platform: String],
        weights: (name: String, checksum: String)? = nil
    ) -> Self {
        let root = "https://huggingface.co/\(source.repository)/resolve/\(source.revision)"
        let libraries = checksums.map { platform, checksum in
            let architecture = platform == .darwinARM64 ? "arm64" : "x86_64"
            let filename = "cactus_needle-\(version)-py3-none-macosx_11_0_\(architecture).whl"
            return (
                platform,
                EngineArtifact(
                    name: libraryName,
                    url: "\(root)/python/\(filename)",
                    checksum: checksum,
                    archivePath: "needle/\(libraryName)"
                )
            )
        }
        return Self(
            generation: generation,
            version: version,
            libraryName: libraryName,
            libraries: Dictionary(uniqueKeysWithValues: libraries),
            weights: weights.map { EngineArtifact(name: $0.name, url: "\(root)/\($0.name)", checksum: $0.checksum) }
        )
    }

    /// Selects 2 or 3; zero defaults to 3.
    init(_ generation: Int) throws {
        switch generation {
        case 2: self = .needle2
        case 0, 3: self = .needle3
        default: throw NeedleError.unsupportedGeneration("unsupported generation \(generation); want 2 or 3")
        }
    }

    private init(
        generation: Int,
        version: String,
        libraryName: String,
        libraries: [Engine.Platform: EngineArtifact],
        weights: EngineArtifact?
    ) {
        self.generation = generation
        self.version = version
        self.libraryName = libraryName
        self.libraries = libraries
        self.weights = weights
    }

    func library(platform: Engine.Platform) throws -> EngineArtifact {
        guard let library = libraries[platform] else {
            throw NeedleError.download("no \(version) engine build for \(platform.rawValue)")
        }
        return library
    }
}
