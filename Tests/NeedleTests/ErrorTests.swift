import Foundation
@testable import Needle
import Testing

private extension Error {
    var needleCase: String? {
        switch self as? NeedleError {
        case .invalidInput: "invalidInput"
        case .unsupportedGeneration: "unsupportedGeneration"
        case .unsupportedABI: "unsupportedABI"
        case .embeddingUnsupported: "embeddingUnsupported"
        case .speechUnsupported: "speechUnsupported"
        case .nativeFailure: "nativeFailure"
        case .native: "native"
        case .streamActive: "streamActive"
        case .closed: "closed"
        case .download: "download"
        case .validation: "validation"
        case nil: nil
        }
    }
}

@Test
func typedErrorsAreMatchable() async throws {
    #expect { try Session(Configuration(generation: 4)) } throws: { $0.needleCase == "unsupportedGeneration" }
    #expect { try EngineRelease(1) } throws: { $0.needleCase == "unsupportedGeneration" }
    var symbols = nativeSymbols(modern: true)
    symbols["needle_embed"] = nil
    #expect { try NativeAPI.bind { symbols[$0] } } throws: { error in
        guard case .unsupportedABI(missing: "needle_embed") = error as? NeedleError else { return false }
        return error.localizedDescription == "needle: unsupported native ABI: missing needle_embed"
    }

    let flagged = #"{"type":"call","function_calls":[{"name":"x","arguments":{}}],"validation":{"ungrounded":["x.a"]}}"#
    let agent = try await FakeNative([flagged]).agent()
    await #expect { try await agent.run("hello") } throws: { error in
        guard case let .validation(response) = error as? NeedleError else { return false }
        return response.validation?.ungrounded == ["x.a"]
    }
    await #expect { try await agent.embed("hello") } throws: { $0.needleCase == "embeddingUnsupported" }

    let broken = try await FakeNative(["native failure"], completeCode: -3).agent()
    await #expect { try await broken.complete("hello") } throws: { error in
        guard case .nativeFailure(operation: "complete", code: -3, detail: "native failure") = error as? NeedleError
        else { return false }
        return error.localizedDescription == "needle: complete failed with code -3: native failure"
    }

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let speech = directory.appendingPathComponent("speech.cact")
    try testWeights(speech: true).write(to: speech)
    let textOnly = WhistleConfiguration(weightsPath: speech.path, libraryPath: "/test/libneedle.dylib")
    await #expect {
        try await Whistle(configuration: textOnly, runtime: NativeRuntime(api: FakeNative().api))
    } throws: { $0.needleCase == "speechUnsupported" }
}
