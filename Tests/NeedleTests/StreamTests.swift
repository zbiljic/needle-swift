import Foundation
@testable import Needle
import Testing

private func isStreamActive(_ error: any Error) -> Bool {
    if case .streamActive = error as? NeedleError {
        true
    } else {
        false
    }
}

private func isClosed(_ error: any Error) -> Bool {
    if case .closed = error as? NeedleError {
        true
    } else {
        false
    }
}

private func isStopped(_ error: any Error) -> Bool {
    error.localizedDescription == "needle: stream is stopped"
}

private func isSpeechUnsupported(_ error: any Error) -> Bool {
    if case .speechUnsupported = error as? NeedleError {
        true
    } else {
        false
    }
}

private struct SpeechFixture {
    let directory: URL
    let probe: SpeechProbe
    let runtime: NativeRuntime
    let config: WhistleConfiguration
    let speech: Whistle

    init(mode: String = "ok", api: ((inout NativeAPI) -> Void)? = nil) async throws {
        directory = try speechDirectory()
        probe = SpeechProbe(mode: mode)
        var native = probe.api
        api?(&native)
        runtime = NativeRuntime(api: native)
        config = WhistleConfiguration(
            weightsPath: directory.appendingPathComponent("speech.cact").path, libraryPath: "/test/libneedle.dylib"
        )
        speech = try await Whistle(configuration: config, runtime: runtime)
    }

    func agent() async throws -> Agent {
        try await Agent(
            configuration: Configuration(
                weightsPath: directory.appendingPathComponent("text.cact").path, libraryPath: "/test/libneedle.dylib"
            ), runtime: runtime
        )
    }

    func cleanup() {
        try? FileManager.default.removeItem(at: directory)
    }
}

@Test
func streamProcessAndStop() async throws {
    let fixture = try await SpeechFixture()
    defer { fixture.cleanup() }
    let (speech, probe, config) = (fixture.speech, fixture.probe, fixture.config)
    let agent = try await fixture.agent()
    let options = AudioOptions(language: "en", keywords: ["Paris", "Ada"], wordTimestamps: true)
    let live = try speech.stream(options: options)
    #expect(probe.streamEvents.isEmpty && probe.text.snapshot.loads == 2)
    let step = try await live.process([0.5, -0.25])
    #expect(step.text == "hello" && step.pending == "there" && step.language == "en")
    #expect(step.received == 1 && step.passMS == 12 && step.words.first?.end == 0.4)
    #expect(try JSONDecoder().decode(StreamTranscript.self, from: JSONEncoder().encode(step)) == step)
    // Other speech work waits for stop; the open stream's own calls proceed.
    await #expect { try await speech.transcribe([]) } throws: { isStreamActive($0) }
    await #expect { try await speech.complete([], agent: agent) } throws: { isStreamActive($0) }
    let second = try speech.stream()
    await #expect { try await second.process([]) } throws: { isStreamActive($0) }
    await #expect { try await second.stop() } throws: { isStreamActive($0) }
    // Switching speech weights would discard the stream.
    let otherPath = fixture.directory.appendingPathComponent("other.cact")
    try testWeights(speech: true).write(to: otherPath)
    var other = config
    other.weightsPath = otherPath.path
    await #expect { try await Whistle(configuration: other, runtime: fixture.runtime) } throws: { isStreamActive($0) }
    #expect(probe.text.snapshot.loads == 2)
    _ = try await live.process([])
    let tail = try await live.stop()
    #expect(tail.text == "there" && tail.received == 2 && tail.words.isEmpty)
    await #expect { try await live.process([]) } throws: { isStopped($0) }
    await #expect { try await live.stop() } throws: { isStopped($0) }
    #expect(try await speech.transcribe([]).text == "Paris")
    #expect(probe.streamEvents == [
        "process/2/0.5/en/Paris,Ada/\(config.bufferSize)",
        "process/0/-/en/Paris,Ada/\(config.bufferSize)",
        "stop/\(config.bufferSize)",
    ])
}

@Test(arguments: ["error", "malformed"])
func streamFailureEndsStream(mode: String) async throws {
    let fixture = try await SpeechFixture(mode: mode)
    defer { fixture.cleanup() }
    let live = try fixture.speech.stream()
    await #expect { try await live.process([]) } throws: { error in
        if case .nativeFailure(operation: "stream", code: -3, detail: "cleared") = error as? NeedleError {
            return mode == "error"
        }
        return mode == "malformed" && error is DecodingError
    }
    let events = fixture.probe.streamEvents
    #expect(events.count == 2 && events[1].hasPrefix("stop/"))
    await #expect { try await live.process([]) } throws: { isStopped($0) }
    // The engine is free again; the probe still fails every call in this mode.
    await #expect { try await fixture.speech.transcribe([]) } throws: { !isStreamActive($0) }
}

@Test
func streamRejectsInvalidInputAndOlderEngines() async throws {
    let older = try await SpeechFixture { api in
        api.streamProcess = nil
        api.streamStop = nil
    }
    defer { older.cleanup() }
    for options in [AudioOptions(language: "sr"), AudioOptions(keywords: ["Ada\0"])] {
        #expect(throws: NeedleError.self) { try older.speech.stream(options: options) }
    }
    // Engine 3.1 has speech without streaming.
    let unsupported = try older.speech.stream()
    await #expect { try await unsupported.process([]) } throws: { isSpeechUnsupported($0) }
    await #expect { try await unsupported.stop() } throws: { isSpeechUnsupported($0) }

    let fixture = try await SpeechFixture()
    defer { fixture.cleanup() }
    let live = try fixture.speech.stream()
    for pcm: [Float] in [[.nan], [1.5]] {
        await #expect(throws: NeedleError.self) { try await live.process(pcm) }
    }
    let cancelledProcess = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await live.process([])
    }
    await #expect(throws: CancellationError.self) { try await cancelledProcess.value }
    let cancelledStop = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await live.stop()
    }
    await #expect(throws: CancellationError.self) { try await cancelledStop.value }
    #expect(fixture.probe.streamEvents.isEmpty)
    #expect(try await fixture.speech.transcribe([]).text == "Paris")
}

@Test
func streamGrowsBufferAndCloseEndsStream() async throws {
    let fixture = try await SpeechFixture()
    defer { fixture.cleanup() }
    let live = try fixture.speech.stream()
    _ = try await live.process([Float](repeating: 0, count: 100 * Whistle.sampleRate))
    await fixture.speech.close()
    await fixture.speech.close()
    await #expect { try await live.process([]) } throws: { isStopped($0) }
    await #expect { try await fixture.speech.transcribe([]) } throws: { isClosed($0) }
    #expect(fixture.probe.streamEvents == [
        "process/\(100 * Whistle.sampleRate)/0.0///\(100 * Whistle.outputBytesPerSecond)",
        "stop/\(fixture.config.bufferSize)",
    ])
    let other = try await Whistle(configuration: fixture.config, runtime: fixture.runtime)
    #expect(try await other.transcribe([]).text == "Paris")
    #expect(fixture.probe.text.snapshot.loads == 1)
}

@Test
func closeReleasesAgentAndWhistle() async throws {
    let fixture = try await SpeechFixture()
    defer { fixture.cleanup() }
    let agent = try await fixture.agent()
    #expect(fixture.probe.text.snapshot.systems.count == 1)
    await agent.close()
    await agent.close()
    await #expect { try await agent.complete("hello") } throws: { isClosed($0) }
    await #expect { try await agent.run("hello") } throws: { isClosed($0) }
    await #expect { try await agent.embed("hello") } throws: { isClosed($0) }
    await #expect { try await agent.reset() } throws: { isClosed($0) }
    #expect(fixture.probe.text.snapshot.inputs.isEmpty && fixture.probe.text.snapshot.resets == 0)
    // The next agent reinitializes the native session.
    let other = try await fixture.agent()
    try await other.reset()
    #expect(fixture.probe.text.snapshot.systems.count == 2 && fixture.probe.text.snapshot.resets == 1)

    await fixture.speech.close()
    await #expect { try await fixture.speech.transcribe([]) } throws: { isClosed($0) }
    await #expect { try await fixture.speech.complete([], agent: other) } throws: { isClosed($0) }
    #expect(fixture.probe.callCount == 0 && fixture.probe.text.snapshot.loads == 2)
}
