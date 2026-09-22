import CryptoKit
import Foundation

/// Downloads the same pinned Needle engines and base weights as needle-go.
public enum Engine {
    public static let version = "3.0.1"
    static let maximumArtifactSize = 64 * 1024 * 1024

    public enum Platform: String, CaseIterable, Sendable {
        case darwinARM64 = "darwin-arm64"
        case darwinAMD64 = "darwin-amd64"

        public static var current: Self {
            #if arch(arm64)
                .darwinARM64
            #else
                .darwinAMD64
            #endif
        }
    }

    /// Returns a trusted cached library without downloading. The directory is an exact override.
    public static func cached(
        generation: Int = 3, platform: Platform = .current, cacheDirectory: URL? = nil
    ) throws -> URL {
        let release = try EngineRelease(generation)
        let target = try directory(release: release, platform: platform, override: cacheDirectory)
            .appendingPathComponent(release.libraryName)
        guard try target.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
            throw NeedleError.download("cached engine is not a regular file: \(target.path)")
        }
        return target
    }

    /// Fetches and verifies the library and required base weights for offline use.
    /// `session` allows callers to configure proxies, timeouts, and URL loading.
    public static func fetch(
        generation: Int = 3,
        platform: Platform = .current,
        cacheDirectory: URL? = nil,
        session: URLSession = .shared
    ) async throws -> URL {
        let library = try await fetchLibrary(
            generation: generation, platform: platform, cacheDirectory: cacheDirectory, session: session
        )
        if try EngineRelease(generation).generation == 3 {
            _ = try await fetchBaseWeights(platform: platform, cacheDirectory: cacheDirectory, session: session)
        }
        return library
    }

    public static func version(for generation: Int) throws -> String {
        try EngineRelease(generation).version
    }

    /// Reads only the .cact header; does not validate the remaining archive.
    public static func weightsGeneration(at path: String) throws -> Int {
        try validateCString(path, name: "weights path")
        let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? file.close() }
        return try NativeRuntime.weightsGeneration(file.read(upToCount: 4) ?? Data())
    }

    static func fetchLibrary(
        generation: Int, platform: Platform = .current, cacheDirectory: URL?, session: URLSession = .shared
    ) async throws -> URL {
        let release = try EngineRelease(generation)
        return try await EngineDownloads.shared.fetch(
            artifact: release.library(platform: platform),
            directory: directory(release: release, platform: platform, override: cacheDirectory),
            session: session
        )
    }

    static func fetchBaseWeights(
        platform: Platform = .current, cacheDirectory: URL?, session: URLSession = .shared
    ) async throws -> URL {
        let release = try EngineRelease(3)
        return try await EngineDownloads.shared.fetch(
            artifact: release.baseWeights,
            directory: directory(release: release, platform: platform, override: cacheDirectory),
            session: session
        )
    }

    static func directory(release: EngineRelease, platform: Platform, override: URL?) throws -> URL {
        if let override {
            guard override.isFileURL else { throw NeedleError.invalidInput("cache directory must be a file URL") }
            return override.standardizedFileURL
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/cactus-needle/\(release.version)/\(platform.rawValue)", isDirectory: true)
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func verifiedCache(_ target: URL, checksum: String) -> Bool {
        guard
            let data = try? Data(contentsOf: target), !data.isEmpty,
            let marker = try? String(contentsOf: target.appendingPathExtension("sha256"), encoding: .utf8)
        else { return false }
        return marker == "\(checksum)\n\(digest(data))\n"
    }

    static func download(artifact: EngineArtifact, directory: URL, session: URLSession) async throws -> URL {
        try Task.checkCancellation()
        let target = directory.appendingPathComponent(artifact.name)
        if verifiedCache(target, checksum: artifact.checksum) {
            return target
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        guard let url = URL(string: artifact.url) else { throw NeedleError.download("invalid artifact URL") }
        var request = URLRequest(url: url, timeoutInterval: 300)
        request.setValue("needle-swift/\(version)", forHTTPHeaderField: "User-Agent")
        var downloaded = Data()
        for attempt in 0 ..< 3 {
            do {
                if attempt > 0 {
                    try await Task.sleep(for: .milliseconds(250 * attempt))
                }
                downloaded = try await downloadAttempt(request, checksum: artifact.checksum, session: session)
                break
            } catch {
                try Task.checkCancellation()
                if attempt == 2 {
                    throw error
                }
            }
        }
        try Task.checkCancellation()
        let installed: Data
        if let member = artifact.archivePath {
            let temporary = directory.appendingPathComponent(".needle-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let archive = temporary.appendingPathComponent("engine.whl")
            try downloaded.write(to: archive)
            installed = try extractLibrary(archive: archive, temporary: temporary, member: member)
        } else {
            installed = downloaded
        }
        try Task.checkCancellation()
        try installed.write(to: target, options: .atomic)
        if artifact.archivePath != nil {
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: target.path)
        }
        let marker = "\(artifact.checksum)\n\(digest(installed))\n"
        try marker.write(to: target.appendingPathExtension("sha256"), atomically: true, encoding: .utf8)
        return target
    }

    static func downloadAttempt(_ request: URLRequest, checksum: String, session: URLSession) async throws -> Data {
        let (bytes, response) = try await session.bytes(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw NeedleError
                .download("engine download returned HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        guard response.expectedContentLength <= maximumArtifactSize else {
            throw NeedleError.download("artifact exceeds \(maximumArtifactSize) bytes")
        }
        var data = Data()
        for try await byte in bytes {
            guard data.count < maximumArtifactSize else {
                throw NeedleError.download("artifact exceeds \(maximumArtifactSize) bytes")
            }
            data.append(byte)
        }
        guard digest(data) == checksum else { throw NeedleError.download("engine checksum mismatch") }
        return data
    }

    static func extractLibrary(archive: URL, temporary: URL, member: String) throws -> Data {
        let output = temporary.appendingPathComponent("libneedle.dylib")
        guard FileManager.default.createFile(atPath: output.path, contents: nil) else {
            throw NeedleError.download("cannot create temporary library")
        }
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        // Only called after checking the pinned wheel's hash; stream one exact member, never unpack paths.
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-p", archive.path, member]
        process.standardOutput = handle
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw NeedleError.download("cannot extract engine library") }
        let size = try output.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0, size <= maximumArtifactSize else { throw NeedleError.download("invalid engine library size") }
        return try Data(contentsOf: output)
    }
}

private actor EngineDownloads {
    static let shared = EngineDownloads()
    private var pending: [URL: (id: UUID, task: Task<URL, Error>)] = [:]

    func fetch(artifact: EngineArtifact, directory: URL, session: URLSession) async throws -> URL {
        // Serialize installs into the same directory, including requests for different architectures.
        let previous = pending[directory]?.task
        let id = UUID()
        let task = Task {
            _ = try? await previous?.value
            try Task.checkCancellation()
            return try await Engine.download(artifact: artifact, directory: directory, session: session)
        }
        pending[directory] = (id, task)
        defer {
            if pending[directory]?.id == id {
                pending[directory] = nil
            }
        }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

struct EngineArtifact: Sendable {
    let name: String
    let url: String
    let checksum: String
    var archivePath: String?
}

struct EngineRelease {
    let generation: Int

    init(_ generation: Int) throws {
        guard [0, 2, 3].contains(generation) else {
            throw NeedleError.invalidInput("unsupported generation \(generation); want 2 or 3")
        }
        self.generation = generation == 0 ? 3 : generation
    }

    var version: String {
        generation == 2 ? "2.0.4" : Engine.version
    }

    var libraryName: String {
        generation == 2 ? "libneedle.dylib" : "libneedle3.dylib"
    }

    var rootURL: String {
        let revision = generation == 2
            ? "32e9e3a93b205f786929697446ae669cf0a84579"
            : "9da75122d4ca11aa4a667281c9c8ba38a7eed679"
        return "https://huggingface.co/Cactus-Compute/needle\(generation)/resolve/\(revision)"
    }

    func library(platform: Engine.Platform) -> EngineArtifact {
        let architecture = platform == .darwinARM64 ? "arm64" : "x86_64"
        let filename = "cactus_needle-\(version)-py3-none-macosx_11_0_\(architecture).whl"
        let checksum = switch (generation, platform) {
        case (2, .darwinARM64): "abae4cca0a4d84ec73da4bde18803b9be812a9209e48fb7fa372002ebaa60265"
        case (2, .darwinAMD64): "071e93d996021b4f6b5bee055777cd81be9f6963d2565115ec461b9f71c7245e"
        case (_, .darwinARM64): "9d3ba55986ad664ddffac4aec680c0657dc82ad04f868898e8085f13755c025f"
        case (_, .darwinAMD64): "22d2693ea23439c2934556c2da8d0aaf708d55546a8b9b68a0647333b42501eb"
        }
        return EngineArtifact(
            name: libraryName,
            url: "\(rootURL)/python/\(filename)",
            checksum: checksum,
            archivePath: "needle/\(libraryName)"
        )
    }

    var baseWeights: EngineArtifact {
        EngineArtifact(
            name: "needle3.cact",
            url: "\(rootURL)/needle3.cact",
            checksum: "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38"
        )
    }
}
