import Foundation
@testable import Needle
import Testing

private final class SpeechProbe: @unchecked Sendable {
    let text = FakeNative()
    let mode: String
    private let lock = NSLock()
    private var settings: [AudioOptions] = []
    private var calls = 0

    init(mode: String = "ok") {
        self.mode = mode
    }

    var options: [AudioOptions] {
        lock.withLock { settings }
    }

    var callCount: Int {
        lock.withLock { calls }
    }

    var api: NativeAPI {
        var api = text.api
        api.setAudio = { value in self.lock.withLock { self.settings.append(value) } }
        api.lastError = { self.options.last?.language == "en" ? "audio diagnostic" : "cleared" }
        api.transcribe = { pcm, _, output in
            #expect(pcm.baseAddress != nil)
            return self.write(
                #"""
                {"text":"Paris","language":"fr","ttft_ms":2,"decode_tps":3,
                 "words":[{"word":"Paris","start":0,"end":0.5,"probability":0.9}]}
                """#,
                to: output
            )
        }
        api.completeAudio = { pcm, tokens, output in
            #expect(pcm.baseAddress != nil)
            #expect(tokens == 512)
            return self.write(
                #"""
                {"type":"call","confidence":0.9,"function_calls":[],"audio_text":"Paris",
                 "audio_language":"fr","audio_ttft_ms":2,"audio_decode_tps":3}
                """#,
                to: output
            )
        }
        return api
    }

    private func write(_ json: String, to output: UnsafeMutableBufferPointer<CChar>) -> Int32 {
        lock.withLock { calls += 1 }
        if mode == "error" {
            return -3
        }
        if mode == "cancel" {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        let value = mode == "malformed" ? "invalid" : json
        for (index, byte) in value.utf8.prefix(output.count).enumerated() {
            output[index] = CChar(bitPattern: byte)
        }
        return 0
    }
}

private func speechDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try testWeights(speech: true).write(to: directory.appendingPathComponent("speech.cact"))
    try testWeights(heads: [0x4000]).write(to: directory.appendingPathComponent("text.cact"))
    return directory
}

@Test
func speechPreservesTextAndRejectsWrongKind() async throws {
    let directory = try speechDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let probe = SpeechProbe()
    let runtime = NativeRuntime(api: probe.api)
    let config = WhistleConfiguration(
        weightsPath: directory.appendingPathComponent("speech.cact").path, libraryPath: "/test/libneedle.dylib"
    )
    let speech = try await Whistle(configuration: config, runtime: runtime)
    #expect(probe.text.snapshot.systems.isEmpty)
    let agent = try await Agent(
        configuration: Configuration(
            stateless: true,
            weightsPath: directory.appendingPathComponent("text.cact").path,
            libraryPath: "/test/libneedle.dylib"
        ), runtime: runtime
    )
    let options = AudioOptions(language: "en", keywords: ["Paris", "Ada"], wordTimestamps: true)
    let transcript = try await speech.transcribe([], options: options)
    #expect(transcript.words?.first?.word == "Paris")
    #expect(transcript.ttftMS == 2)
    #expect(try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(transcript)) == transcript)
    let response = try await speech.complete([], agent: agent, options: options)
    #expect(response.audioText == "Paris" && response.audioLanguage == "fr")
    #expect(response.audioTTFTMS == 2 && response.audioDecodeTPS == 3)
    #expect(response.confidence == 0.9)
    #expect(try JSONDecoder().decode(Response.self, from: JSONEncoder().encode(response)) == response)
    #expect(probe.options == [options, AudioOptions()])
    #expect(probe.text.snapshot.resets == 1)
    let otherPath = directory.appendingPathComponent("other.cact")
    try testWeights(speech: true).write(to: otherPath)
    var other = config
    other.weightsPath = otherPath.path
    let second = try await Whistle(configuration: other, runtime: runtime)
    _ = try await second.transcribe([])
    _ = try await speech.transcribe([])
    #expect(probe.text.snapshot.systems.count == 1)
    #expect(probe.text.snapshot.loads == 4)
    other.weightsPath = agent.session.weightsPath
    await #expect(throws: NeedleError.self) { try await Whistle(configuration: other, runtime: runtime) }
    let wrong = try Session(Configuration(weightsPath: speech.session.weightsPath))
    await #expect(throws: NeedleError.self) {
        try await runtime.initialize(wrong, library: URL(fileURLWithPath: "/test/libneedle.dylib"))
    }
    #expect(probe.text.snapshot.loads == 4)
}

@Test
func speechInputValidationPrecedesNativeCalls() async throws {
    let directory = try speechDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let probe = SpeechProbe()
    let runtime = NativeRuntime(api: probe.api)
    let config = WhistleConfiguration(
        weightsPath: directory.appendingPathComponent("speech.cact").path, libraryPath: "/test/libneedle.dylib"
    )
    let speech = try await Whistle(configuration: config, runtime: runtime)
    for pcm: [Float] in [
        [.nan],
        [.infinity],
        [-.infinity],
        [1.01],
        [-1.01],
        [Float](repeating: 0, count: Whistle.maximumSamples + 1),
    ] {
        await #expect(throws: NeedleError.self) { try await speech.transcribe(pcm) }
    }
    for options in [AudioOptions(language: "xx"), AudioOptions(keywords: ["bad\0value"])] {
        await #expect(throws: NeedleError.self) { try await speech.transcribe([], options: options) }
    }
    for size in [0, 1, Int(Int32.max) + 1] {
        var invalid = config
        invalid.bufferSize = size
        await #expect(throws: NeedleError.self) { try await Whistle(configuration: invalid, runtime: runtime) }
    }
    let legacy = try await probe.text.agent()
    await #expect(throws: NeedleError.self) { try await speech.complete([], agent: legacy) }
    await #expect(throws: NeedleError.self) {
        try await Whistle(configuration: config, runtime: NativeRuntime(api: probe.text.api))
    }
    let cancelled = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await speech.transcribe([])
    }
    await #expect(throws: CancellationError.self) { try await cancelled.value }
    #expect(probe.callCount == 0)
    try AudioOptions().validate([-1, 0, 1])
}

@Test(arguments: ["error", "malformed", "cancel"])
func audioCleanupSurvivesFailures(mode: String) async throws {
    let directory = try speechDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let probe = SpeechProbe(mode: mode)
    let runtime = NativeRuntime(api: probe.api)
    let speech = try await Whistle(
        configuration: WhistleConfiguration(
            weightsPath: directory.appendingPathComponent("speech.cact").path, libraryPath: "/test/libneedle.dylib"
        ), runtime: runtime
    )
    let agent = try await Agent(
        configuration: Configuration(
            weightsPath: directory.appendingPathComponent("text.cact").path, libraryPath: "/test/libneedle.dylib"
        ), runtime: runtime
    )
    for tokens in [-1, Int(Int32.max) + 1] {
        await #expect(throws: NeedleError.self) { try await speech.complete([], agent: agent, maxNewTokens: tokens) }
    }
    #expect(probe.callCount == 0)
    let task = Task { try await speech.complete([], agent: agent, options: AudioOptions(language: "en")) }
    do {
        _ = try await task.value
        Issue.record("Expected audio failure")
    } catch {
        if mode == "error" {
            #expect(error.localizedDescription.contains("audio diagnostic"))
        }
        if mode == "cancel" {
            #expect(error is CancellationError)
        }
    }
    #expect(probe.options == [AudioOptions(language: "en"), AudioOptions()])
    #expect(probe.text.snapshot.resets == 0)
}

@Test
func speechCABIArguments() throws {
    let transcribe: NativeAPI.Transcribe = { pcm, count, language, keywords, times, output, capacity in
        guard
            let pcm, pcm[0] == 0.25, count == 2, let language, String(cString: language) == "en",
            let keywords, String(cString: keywords) == "Paris\nAda", times == 1, capacity == 32 else { return -1 }
        output?[0] = 65
        return 7
    }
    let complete: NativeAPI.CompleteWithAudio = { text, pcm, count, tokens, output, capacity in
        guard text == nil, let pcm, pcm[0] == 0.25, count == 2, tokens == 123, capacity == 32 else { return -1 }
        output?[0] = 66
        return 8
    }
    let setAudio: NativeAPI.SetAudio = { language, keywords, times in
        if times == 1 {
            #expect(language.map { String(cString: $0) } == "en")
            #expect(keywords.map { String(cString: $0) } == "Paris\nAda")
        } else {
            #expect(language == nil && keywords == nil)
        }
    }
    var symbols = nativeSymbols(modern: true)
    symbols["needle_transcribe"] = address(transcribe)
    symbols["needle_set_audio"] = address(setAudio)
    symbols["needle_complete"] = address(complete)
    let api = try NativeAPI.bind { symbols[$0] }
    let options = AudioOptions(language: "en", keywords: ["Paris", "Ada"], wordTimestamps: true)
    api.setAudio?(options)
    var buffer = [CChar](repeating: 0, count: 32)
    [Float(0.25), -0.5].withUnsafeBufferPointer { pcm in
        #expect(buffer.withUnsafeMutableBufferPointer { api.transcribe?(pcm, options, $0) } == 7)
        #expect(buffer[0] == 65)
        #expect(buffer.withUnsafeMutableBufferPointer { api.completeAudio?(pcm, 123, $0) } == 8)
        #expect(buffer[0] == 66)
    }
    api.setAudio?(AudioOptions())
}

extension NativeIntegrationTests {
    @Test
    func nativeWhistleSilenceAndWeightKinds() async throws {
        let speech = try await Whistle()
        let agent = try await Agent(configuration: Configuration(stateless: true))
        for pcm: [Float] in [[], [Float](repeating: 0, count: Whistle.sampleRate)] {
            let transcript = try await speech.transcribe(pcm, options: AudioOptions(wordTimestamps: true))
            #expect(transcript.text.isEmpty && transcript.language.isEmpty)
            #expect(transcript.words?.isEmpty != false)
            let response = try await speech.complete(
                pcm, agent: agent, options: AudioOptions(language: "en", wordTimestamps: true)
            )
            #expect(response.success == true)
            #expect(response.audioText?.isEmpty == true && response.audioLanguage != nil)
            #expect(response.audioTTFTMS != nil && response.audioDecodeTPS != nil)
        }
        await #expect(throws: NeedleError.self) {
            try await Agent(configuration: Configuration(weightsPath: speech.session.weightsPath))
        }
        await #expect(throws: NeedleError.self) {
            try await Whistle(configuration: WhistleConfiguration(weightsPath: agent.session.weightsPath))
        }
        let other = try await Agent(configuration: Configuration(system: "device: test"))
        _ = try await other.complete("hello")
        #expect(try await speech.transcribe([]).text.isEmpty)
    }

    @Test
    func nativeSpeechSwitchPreservesTextContinuation() async throws {
        let speech = try await Whistle()
        let agent = try await Agent(configuration: Configuration(tools: [Tool(schema: weatherSchema)]))
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).cact")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: speech.session.weightsPath), to: path)
        defer { try? FileManager.default.removeItem(at: path) }
        var baseline: Response?
        for interrupt in [false, true] {
            try await agent.reset()
            let call = try await agent.complete("What is the weather in Lagos?")
            #expect(call.functionCalls?.first?.name == "get_weather")
            if interrupt {
                let other = try await Whistle(configuration: WhistleConfiguration(weightsPath: path.path))
                _ = try await other.transcribe([])
            }
            let response = try await agent.complete(#"["Clear in Lagos"]"#)
            #expect(response.type == "respond")
            if let baseline {
                #expect(response.functionCalls == baseline.functionCalls)
                #expect(response.reasoning == baseline.reasoning)
                #expect(response.confidence == baseline.confidence)
            }
            baseline = response
        }
    }
}
