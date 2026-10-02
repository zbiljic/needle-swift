import Foundation
@testable import Needle
import Testing

private func address(_ function: some Any) -> UnsafeMutableRawPointer {
    unsafeBitCast(function, to: UnsafeMutableRawPointer.self)
}

@Test(arguments: [false, true])
func nativeCompletionABI(modern: Bool) throws {
    let initialize: NativeAPI.Initialize = { _, _, _ in 12 }
    let reset: NativeAPI.Reset = {}
    let load: NativeAPI.Load = { _, size in size == 4 ? 1 : -1 }
    let legacy: NativeAPI.Complete = { text, tokens, output, capacity in
        guard let text, String(cString: text) == "hello", tokens == 123, capacity == 32 else { return -1 }
        output?[0] = 65
        return 7
    }
    let audio: NativeAPI.CompleteWithAudio = { text, pcm, samples, tokens, output, capacity in
        guard
            let text, String(cString: text) == "hello", pcm == nil, samples == 0,
            tokens == 123, capacity == 32 else { return -1 }
        output?[0] = 65
        return 7
    }
    let lastError: @convention(c) () -> UnsafePointer<CChar>? = {
        let message: StaticString = "native detail"
        return UnsafeRawPointer(message.utf8Start).assumingMemoryBound(to: CChar.self)
    }
    var symbols = [
        "needle_init": address(initialize), "needle_reset": address(reset), "needle_load": address(load),
        "needle_complete": modern ? address(audio) : address(legacy),
    ]
    if modern {
        for name in ["needle_models", "needle_transcribe", "needle_set_audio", "needle_embed"] {
            symbols[name] = address(reset) // Capability probes only; these are not called here.
        }
        symbols["needle_last_error"] = address(lastError)
    }
    let api = try NativeAPI.bind { symbols[$0] }
    var buffer = [CChar](repeating: 0, count: 32)
    #expect(buffer.withUnsafeMutableBufferPointer { api.complete("hello", 123, $0) } == 7)
    #expect(buffer[0] == 65)
    #expect(api.lastError?() == (modern ? "native detail" : nil))
    #expect(api.failure("test", code: -2, fallback: "buffer").localizedDescription
        == "needle: test failed with code -2: \(modern ? "native detail" : "buffer")")
    for name in Array(symbols.keys) {
        let removed = symbols.removeValue(forKey: name)
        #expect(throws: NeedleError.self) { try NativeAPI.bind { symbols[$0] } }
        symbols[name] = removed
    }
}

@Test
func nativeDiagnosticFallbacks() {
    var api = NativeAPI(initialize: { _, _, _ in 0 }, complete: { _, _, _ in 0 }, reset: {}, load: { _, _ in 0 })
    for detail: String? in [nil, "", " \n"] {
        api.lastError = { detail }
        #expect(api.failure("load", code: -1).localizedDescription == "needle: load failed with code -1")
        #expect(api.failure("complete", code: -1, fallback: " buffer \n").localizedDescription
            == "needle: complete failed with code -1: buffer")
    }
}
