import Foundation
@testable import Needle
import Testing

@Test
func embeddingBoundsErrorsAndStatelessContext() async throws {
    let session = try Session(Configuration(stateless: true))
    let library = URL(fileURLWithPath: "/test/libneedle.dylib")
    let fake = FakeNative()
    var api = fake.api
    api.embed = { text, buffer in
        #expect(text == "hello")
        if buffer.isEmpty {
            return 2
        }
        buffer[0] = 0.5
        buffer[1] = -0.5
        return 2
    }
    let runtime = NativeRuntime(api: api)
    try await runtime.initialize(session, library: library)
    #expect(try await runtime.embed(session, text: "hello") == [0.5, -0.5])
    #expect(fake.snapshot.resets == 0)
    await #expect(throws: NeedleError.self) { try await runtime.embed(session, text: "bad\0input") }
    let legacy = try Session(Configuration(generation: 2))
    await #expect(throws: NeedleError.self) { try await runtime.embed(legacy, text: "hello") }
    let missing = NativeRuntime(api: fake.api)
    try await missing.initialize(session, library: library)
    await #expect(throws: NeedleError.self) { try await missing.embed(session, text: "hello") }
    for (size, result): (Int32, Int32) in [(0, 0), (-1, 0), (1 << 20 + 1, 0), (2, -2), (2, 1)] {
        api.embed = { _, buffer in buffer.isEmpty ? size : result }
        api.lastError = { "embedding failed" }
        let runtime = NativeRuntime(api: api)
        try await runtime.initialize(session, library: library)
        await #expect(throws: NeedleError.self) { try await runtime.embed(session, text: "hello") }
    }
    api.embed = { _, buffer in
        #expect(buffer.isEmpty)
        withUnsafeCurrentTask { $0?.cancel() }
        return 2
    }
    let cancelledRuntime = NativeRuntime(api: api)
    try await cancelledRuntime.initialize(session, library: library)
    let cancelled = Task { try await cancelledRuntime.embed(session, text: "hello") }
    await #expect(throws: CancellationError.self) { try await cancelled.value }
}
