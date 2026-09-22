import Foundation
@testable import Needle
import Testing

private struct WeatherArguments: Codable, Sendable, Equatable {
    let city: String
}

private let weatherSchema = ToolSchema(
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: [
        "type": .string("object"),
        "properties": .object(["city": .object(["type": .string("string")])]),
        "required": .array([.string("city")]),
    ]
)

/// Test state is shared by @Sendable native callbacks and assertions, always under the lock.
private final class FakeNative: @unchecked Sendable {
    private let lock = NSLock()
    private var responses: [String]
    private var inputs: [String] = []
    private var systems: [String] = []
    private var resetCount = 0
    private var loadCount = 0
    private var completeCode: Int32
    private let loadCode: Int32

    init(_ responses: [String] = [], completeCode: Int32 = 0, loadCode: Int32 = 0) {
        self.responses = responses
        self.completeCode = completeCode
        self.loadCode = loadCode
    }

    // swiftlint:disable:next large_tuple
    var snapshot: (inputs: [String], systems: [String], resets: Int, loads: Int) {
        lock.withLock { (inputs, systems, resetCount, loadCount) }
    }

    var api: NativeAPI {
        NativeAPI(
            initialize: { system, _, _ in
                self.lock.withLock { self.systems.append(String(cString: system)) }
                return 0
            },
            complete: { input, _, buffer in
                self.lock.withLock {
                    self.inputs.append(input)
                    let response = self.responses.isEmpty ? "" : self.responses.removeFirst()
                    for (index, byte) in response.utf8.prefix(buffer.count).enumerated() {
                        buffer[index] = CChar(bitPattern: byte)
                    }
                    return self.completeCode
                }
            },
            reset: { self.lock.withLock { self.resetCount += 1 } },
            load: { _, _ in
                self.lock.withLock { self.loadCount += 1 }
                return self.loadCode
            }
        )
    }

    func agent(tools: [Tool] = [], bufferSize: Int = 65536) async throws -> Agent {
        try await Agent(
            configuration: Configuration(
                tools: tools,
                generation: 2,
                bufferSize: bufferSize,
                libraryPath: "/test/libneedle.dylib"
            ),
            runtime: NativeRuntime(api: api)
        )
    }
}

private let weatherCall = #"{"type":"call","function_calls":[{"name":"get_weather","arguments":{"city":"Lagos"}}]}"#
private let finalResponse = #"{"type":"respond","confidence":0.94,"prefill_tps":12,"decode_tps":4,"peak_ram_mb":30}"#

@Test
func typedToolLoopAndExtraction() async throws {
    let tool = Tool(schema: weatherSchema) { (arguments: WeatherArguments) in "Clear in \(arguments.city)" }
    let fake = FakeNative([weatherCall, finalResponse])
    let agent = try await fake.agent(tools: [tool])
    let response = try await agent.run("What is the weather in Lagos?")
    #expect(response.type == "respond")
    #expect(response.results == [.string("Clear in Lagos")])
    #expect(response.prefillTPS == 12)
    #expect(response.peakRAMMB == 30)
    let submitted = try JSONDecoder().decode([String].self, from: Data(fake.snapshot.inputs[1].utf8))
    #expect(submitted == ["Clear in Lagos"])
    let call = try JSONDecoder().decode(Response.self, from: Data(weatherCall.utf8))
    let extracted: WeatherArguments = try call.extract()
    #expect(extracted.city == "Lagos")
    #expect(throws: NeedleError.self) { try response.extract(WeatherArguments.self) }
    try await agent.reset()
    #expect(fake.snapshot.resets == 1)
}

private actor InvocationCount {
    var value = 0
    func increment() {
        value += 1
    }
}

@Test
func rejectsEntireFlaggedTurnAndKeepsEarlierResults() async throws {
    let invocations = InvocationCount()
    let tool = Tool(schema: weatherSchema) { (arguments: WeatherArguments) in
        await invocations.increment()
        return arguments.city
    }
    var flagged = try JSONDecoder().decode(Response.self, from: Data(weatherCall.utf8))
    flagged.functionCalls?.append(FunctionCall(name: "get_weather", arguments: .object([:])))
    flagged.validation = Validation(ungrounded: ["get_weather.city"])
    let rejected = try String(decoding: JSONEncoder().encode(flagged), as: UTF8.self)
    let fake = FakeNative([weatherCall, rejected, finalResponse])
    let agent = try await fake.agent(tools: [tool])
    do {
        _ = try await agent.run("weather")
        Issue.record("Expected validation rejection")
    } catch let NeedleError.validation(response) {
        #expect(response.results == [.string("Lagos")])
        #expect(response.functionCalls?.count == 2)
        #expect(fake.snapshot.inputs.count == 2)
    }
    #expect(await invocations.value == 1)
    let initial = try await FakeNative([rejected]).agent(tools: [tool])
    await #expect(throws: NeedleError.self) { try await initial.run("weather") }
    #expect(await invocations.value == 1)
    // Complete and extract deliberately leave validation policy to their caller.
    let raw = try await FakeNative([rejected]).agent()
    #expect(try await raw.complete("weather").validation?.ungrounded == ["get_weather.city"])
    #expect(throws: NeedleError.self) {
        try Response(type: "respond", validation: Validation(negation: true)).validate()
    }
    try Response(type: "respond").validate()
}

@Test
func toolFailuresMissingArgumentsAndStepLimit() async throws {
    let call = #"{"type":"call","function_calls":[{"name":"missing"},{"name":"fails","arguments":null}]}"#
    let tool = Tool(schema: ToolSchema(name: "fails")) { (arguments: JSONValue) async throws -> JSONValue in
        #expect(arguments == .object([:]))
        throw NeedleError.invalidInput("offline")
    }
    let fake = FakeNative([call, call, finalResponse])
    let agent = try await fake.agent(tools: [tool])
    let response = try await agent.run("go", maxSteps: 1)
    #expect(response.type == "call")
    #expect(response.results == [
        .object(["error": .string("unknown tool: missing")]),
        .object(["error": .string("needle: offline")]),
    ])
    #expect(fake.snapshot.inputs.count == 2)
}

@Test
func jsonRoundTripPreservesLargeIntegers() throws {
    let json = #"{"signed":9223372036854775807,"unsigned":18446744073709551615,"bool":true,"array":[null,1.5,"x"]}"#
    let value = try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
    #expect(try JSONValue(value) == value)
    guard case let .object(object) = value else { Issue.record("Expected object"); return }
    #expect(object["signed"] == .integer(Int64.max))
    #expect(object["unsigned"] == .unsignedInteger(UInt64.max))
    #expect(throws: (any Error).self) { try JSONEncoder().encode(JSONValue.number(.infinity)) }
}

@Test
func invalidConfigurationFailsBeforeLoading() throws {
    let duplicate = Tool(schema: ToolSchema(name: "same"))
    for config in [
        Configuration(tools: [Tool(schema: ToolSchema(name: ""))]),
        Configuration(tools: [duplicate, duplicate]),
        Configuration(system: "bad\0system"),
        Configuration(weightsPath: "bad\0weights"),
        Configuration(toolIndexPath: "bad\0index"),
        Configuration(generation: 4),
        Configuration(bufferSize: 1),
        Configuration(bufferSize: Int(Int32.max) + 1),
    ] {
        #expect(throws: NeedleError.self) { try Session(config) }
    }
    #expect(throws: NeedleError.self) { try NativeAPI.open("/nonexistent/needle.dylib") }
    #expect(throws: NeedleError.self) { try NativeAPI.open("/usr/lib/libSystem.B.dylib") }
}

@Test
func nativeDiagnosticsAndBufferBoundaries() async throws {
    let fake = FakeNative(["native error\0ignored"], completeCode: -3)
    let agent = try await fake.agent()
    do {
        _ = try await agent.complete("hello")
        Issue.record("Expected native failure")
    } catch {
        #expect(error.localizedDescription == "needle: complete failed with code -3: native error")
    }
    for output in ["12345678", "invalid", #"{"type":""}"#, #"{"success":true}"#] {
        let invalid = try await FakeNative([output]).agent(bufferSize: output == "12345678" ? 8 : 65536)
        await #expect(throws: (any Error).self) { try await invalid.complete("hello") }
    }
    await #expect(throws: NeedleError.self) { try await agent.complete("bad\0input") }
    await #expect(throws: NeedleError.self) { try await agent.complete("hello", maxNewTokens: -1) }
    await #expect(throws: NeedleError.self) { try await agent.complete("hello", maxNewTokens: Int(Int32.max) + 1) }
    await #expect(throws: NeedleError.self) { try await agent.run("hello", maxSteps: -1) }
    #expect(fake.snapshot.inputs.count == 1)
}

@Test
func cancellationDoesNotBecomeAToolResult() async throws {
    let tool = Tool(schema: weatherSchema) { (_: JSONValue) async throws -> JSONValue in
        throw CancellationError()
    }
    let fake = FakeNative([weatherCall, finalResponse])
    let agent = try await fake.agent(tools: [tool])
    await #expect(throws: CancellationError.self) { try await agent.run("weather") }
    #expect(fake.snapshot.inputs.count == 1)
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await agent.complete("hello")
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(fake.snapshot.inputs.count == 1)
}

@Test
func rebindingAndTunedWeightSafeguards() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let weights = directory.appendingPathComponent("tuned.cact")
    try Data([0x83, 0x2A, 0xE1, 0x05, 1]).write(to: weights)
    for header in [[], [0x83], [1, 2, 3, 4]] as [[UInt8]] {
        #expect(throws: NeedleError.self) { try NativeRuntime.weightsGeneration(Data(header)) }
    }
    let fake = FakeNative([finalResponse])
    let runtime = NativeRuntime(api: fake.api)
    let base = try Session(Configuration(system: "base", generation: 2))
    let other = try Session(Configuration(system: "other", generation: 2))
    let tuned = try Session(Configuration(system: "tuned", weightsPath: weights.path))
    let library = URL(fileURLWithPath: "/test/libneedle.dylib")
    try await runtime.initialize(base, library: library)
    try await runtime.reset(other)
    try await runtime.reset(base)
    #expect(fake.snapshot.systems == ["base", "other", "base"])
    try await runtime.initialize(tuned, library: library)
    let response = try await runtime.complete(tuned, text: "hello", tokens: 32)
    #expect(response.confidence == nil)
    #expect(fake.snapshot.loads == 1)
    await #expect(throws: NeedleError.self) { try await runtime.reset(base) }
    await #expect(throws: NeedleError.self) {
        try await runtime.initialize(tuned, library: URL(fileURLWithPath: "/other/libneedle.dylib"))
    }
}

@Test
func generationSelectionOverridesAndWeightReloads() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    #expect(try Session(Configuration()).generation == 3)
    #expect(try Session(Configuration(generation: 0)).generation == 3)
    #expect(Agent.defaultMaxNewTokens == 512)
    let environment = ["NEEDLE2_LIB_PATH": "/v2", "NEEDLE3_LIB_PATH": "/v3", "NEEDLE_LIB_PATH": "/legacy"]
    for generation in [2, 3] {
        let weights = directory.appendingPathComponent("v\(generation).cact")
        let header = Data([generation == 2 ? 0x83 : 0x84, 0x2A, 0xE1, 0x05, 1])
        try header.write(to: weights)
        #expect(try Engine.weightsGeneration(at: weights.path) == generation)
        var base = try Session(Configuration(generation: generation))
        let tuned = try Session(Configuration(generation: 5 - generation, weightsPath: weights.path))
        #expect(tuned.generation == generation)
        #expect(base.libraryOverride(nil, environment: environment) == "/v\(generation)")
        #expect(base.libraryOverride("/explicit", environment: environment) == "/explicit")
        #expect(base.libraryOverride("", environment: ["NEEDLE_LIB_PATH": "/legacy"])
            == (generation == 2 ? "/legacy" : nil))
        let fake = FakeNative([finalResponse, finalResponse, finalResponse], loadCode: 1)
        let runtime = NativeRuntime(api: fake.api)
        let library = URL(fileURLWithPath: "/test/v\(generation).dylib")
        if generation == 3 {
            let basePath = directory.appendingPathComponent("base.cact")
            try header.write(to: basePath)
            base.weightsPath = basePath.path
        }
        try await runtime.initialize(base, library: library)
        #expect(try await runtime.complete(base, text: "hello", tokens: 512).confidence == 0.94)
        try await runtime.initialize(tuned, library: library)
        #expect(try await runtime.complete(tuned, text: "hello", tokens: 512).confidence == nil)
        if generation == 3 {
            try await runtime.reset(base)
            #expect(try await runtime.complete(base, text: "hello", tokens: 512).confidence == 0.94)
            #expect(fake.snapshot.loads == 3)
        } else {
            await #expect(throws: NeedleError.self) { try await runtime.reset(base) }
        }
        // Recheck the header at load time, before passing a changed file to native code.
        try Data([generation == 2 ? 0x84 : 0x83, 0x2A, 0xE1, 0x05]).write(to: weights)
        let changed = NativeRuntime(api: fake.api)
        let loads = fake.snapshot.loads
        await #expect(throws: NeedleError.self) { try await changed.initialize(tuned, library: library) }
        #expect(fake.snapshot.loads == loads)
        try header.write(to: weights)
        let failed = NativeRuntime(api: FakeNative(loadCode: -7).api)
        await #expect(throws: NeedleError.self) { try await failed.initialize(tuned, library: library) }
    }
}

@Test
func suppressedCallsArePreservedWithoutExecution() async throws {
    let invocations = InvocationCount()
    let tool = Tool(schema: weatherSchema) { (_: JSONValue) in
        await invocations.increment()
        return "unexpected"
    }
    let json = #"""
    {"type":"call","function_calls":[],"suppressed_calls":[{"name":"get_weather","arguments":{"city":"Lagos"}}]}
    """#
    let agent = try await FakeNative([json]).agent(tools: [tool])
    let response = try await agent.run("weather")
    #expect(response.suppressedCalls?.first?.name == "get_weather")
    #expect(response.results?.isEmpty == true)
    #expect(await invocations.value == 0)
    #expect(try JSONDecoder().decode(Response.self, from: JSONEncoder().encode(response)) == response)
}

@Suite(.enabled(if: ProcessInfo.processInfo.environment["NEEDLE_NATIVE_TEST"] == "1"))
struct NativeIntegrationTests {
    @Test
    func downloadsLoadsAndCallsBothGenerations() async throws {
        let tool = Tool(schema: weatherSchema) { (arguments: WeatherArguments) in "Clear in \(arguments.city)" }
        var agents: [Agent] = []
        for generation in [2, 3] {
            let path = try await Engine.fetch(generation: generation)
            #expect(try Engine.cached(generation: generation) == path)
            #expect(try await Engine.fetch(generation: generation) == path)
            try await agents.append(Agent(configuration: Configuration(
                tools: [tool], generation: generation, libraryPath: path.path
            )))
        }
        // Alternate after both libraries are loaded to catch shared-symbol/state collisions.
        for _ in 0 ..< 2 {
            for agent in agents {
                try await agent.reset()
                let response = try await agent.complete("What is the weather in Lagos?")
                #expect(response.type == "call")
                #expect(response.confidence != nil)
                try response.validate()
                let arguments: WeatherArguments = try response.extract()
                #expect(arguments.city == "Lagos")
                try await agent.reset()
                let result = try await agent.run("What is the weather in Lagos?", maxSteps: 2)
                #expect(result.results?.first == .string("Clear in Lagos"))
            }
        }
        // Use a copy of the published v3 archive as custom weights to exercise real reloads.
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let customPath = directory.appendingPathComponent("custom.cact")
        let basePath = try #require(agents[1].session.weightsPath)
        try FileManager.default.copyItem(atPath: basePath, toPath: customPath.path)
        let custom = try await Agent(configuration: Configuration(generation: 2, weightsPath: customPath.path))
        #expect(custom.session.generation == 3)
        #expect(try await custom.complete("hello").confidence == nil)
        try await agents[1].reset()
        #expect(try await agents[1].complete("What is the weather in Lagos?").confidence != nil)
    }
}
