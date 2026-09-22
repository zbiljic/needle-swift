import Foundation
@testable import Needle
import Testing

private final class StubProtocol: URLProtocol, @unchecked Sendable {
    override static func canInit(with _: URLRequest) -> Bool {
        true
    }

    override static func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard
            let url = request.url,
            let response = HTTPURLResponse(
                url: url,
                statusCode: url.lastPathComponent == "status" ? 503 : 200,
                httpVersion: nil,
                headerFields: url.lastPathComponent == "oversize"
                    ? ["Content-Length": "\(Engine.maximumArtifactSize + 1)"] : nil
            )
        else { return }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("verified bytes".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Test
func verifiesDownloadsAndRejectsHTTPSizeAndHashErrors() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let payload = Data("verified bytes".utf8)
    let checksum = Engine.digest(payload)
    let url = try #require(URL(string: "https://needle.test/ok"))
    #expect(try await Engine.downloadAttempt(URLRequest(url: url), checksum: checksum, session: session) == payload)
    for path in ["status", "oversize", "checksum"] {
        let url = try #require(URL(string: "https://needle.test/\(path)"))
        await #expect(throws: NeedleError.self) {
            try await Engine.downloadAttempt(
                URLRequest(url: url), checksum: path == "checksum" ? "wrong" : checksum, session: session
            )
        }
    }
}

@Test
func cacheIntegrityAndFailedDownloadPreserveExistingLibrary() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let target = directory.appendingPathComponent("libneedle.dylib")
    let checksum = try EngineRelease(2).library(platform: .current).checksum
    let data = Data("existing library".utf8)
    try data.write(to: target)
    #expect(try Engine.cached(generation: 2, cacheDirectory: directory) == target)
    #expect(!Engine.verifiedCache(target, checksum: checksum))
    let marker = "\(checksum)\n\(Engine.digest(data))\n"
    try marker.write(to: target.appendingPathExtension("sha256"), atomically: true, encoding: .utf8)
    #expect(Engine.verifiedCache(target, checksum: checksum))
    #expect(!Engine.verifiedCache(target, checksum: "different architecture"))
    try Data("modified".utf8).write(to: target)
    #expect(!Engine.verifiedCache(target, checksum: checksum))
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    await #expect(throws: NeedleError.self) {
        try await Engine.fetch(generation: 2, cacheDirectory: directory, session: session)
    }
    #expect(try Data(contentsOf: target) == Data("modified".utf8))
    #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == [
        "libneedle.dylib", "libneedle.dylib.sha256",
    ])
}

@Test
func extractsOnlyTheLibraryMember() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let source = directory.appendingPathComponent("needle")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let payload = Data("library bytes".utf8)
    try payload.write(to: source.appendingPathComponent("libneedle.dylib"))
    try Data("ignored".utf8).write(to: source.appendingPathComponent("other.txt"))
    let archive = directory.appendingPathComponent("test.whl")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
    process.currentDirectoryURL = directory
    process.arguments = ["-q", archive.path, "needle/libneedle.dylib", "needle/other.txt"]
    try process.run()
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
    #expect(try Engine
        .extractLibrary(archive: archive, temporary: directory, member: "needle/libneedle.dylib") == payload)
    #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent("other.txt").path))
}

@Test
func downloadsAndVerifiesBaseWeightsAndIsolatesGenerationCaches() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let payload = Data("verified bytes".utf8)
    let artifact = EngineArtifact(
        name: "needle3.cact", url: "https://needle.test/ok", checksum: Engine.digest(payload)
    )
    let target = try await Engine.download(artifact: artifact, directory: directory, session: session)
    #expect(try Data(contentsOf: target) == payload)
    #expect(Engine.verifiedCache(target, checksum: artifact.checksum))
    // A valid cache works even with an unreachable URL.
    let cached = EngineArtifact(name: artifact.name, url: "invalid", checksum: artifact.checksum)
    #expect(try await Engine.download(artifact: cached, directory: directory, session: session) == target)
    let bad = EngineArtifact(name: artifact.name, url: artifact.url, checksum: "wrong")
    await #expect(throws: NeedleError.self) {
        try await Engine.download(artifact: bad, directory: directory, session: session)
    }
    #expect(try Data(contentsOf: target) == payload)
    for generation in [2, 3] {
        let release = try EngineRelease(generation)
        let path = directory.appendingPathComponent(release.libraryName)
        try payload.write(to: path)
        #expect(try Engine.cached(generation: generation, cacheDirectory: directory) == path)
        #expect(try Engine.directory(release: release, platform: .current, override: nil).path
            .contains("/\(release.version)/"))
        for platform in Engine.Platform.allCases {
            let library = release.library(platform: platform)
            #expect(library.archivePath == "needle/\(release.libraryName)")
            #expect(library.url.contains("/needle\(generation)/resolve/"))
            #expect(library.url.contains("/python/cactus_needle-\(release.version)-"))
            #expect(library.checksum.count == 64)
        }
    }
    #expect(try Engine.version(for: 0) == "3.0.1")
    #expect(try Engine.version(for: 2) == "2.0.4")
    #expect(try !EngineRelease(3).baseWeights.url.contains("/python/"))
    for generation in [-1, 1, 4] {
        #expect(throws: NeedleError.self) { try Engine.cached(generation: generation) }
        await #expect(throws: NeedleError.self) { try await Engine.fetch(generation: generation) }
    }
}
