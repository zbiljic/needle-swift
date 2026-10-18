import Foundation

/// One step of live transcription. `text` and `words` are the words the step
/// committed, with times in seconds from the start of the stream, and
/// `pending` is the unconfirmed tail. `received` counts the seconds of audio
/// so far and `passMS` the time the step took. Join the texts of consecutive
/// steps with a space.
public struct StreamTranscript: Codable, Sendable, Equatable {
    public var text: String
    public var words: [WordTimestamp]
    public var pending: String
    public var language: String
    public var received: Double
    public var passMS: Double

    enum CodingKeys: String, CodingKey {
        case text, words, pending, language, received
        case passMS = "pass_ms"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decodeIfPresent(String.self, forKey: .text) ?? ""
        words = try container.decodeIfPresent([WordTimestamp].self, forKey: .words) ?? []
        pending = try container.decodeIfPresent(String.self, forKey: .pending) ?? ""
        language = try container.decodeIfPresent(String.self, forKey: .language) ?? ""
        received = try container.decodeIfPresent(Double.self, forKey: .received) ?? 0
        passMS = try container.decodeIfPresent(Double.self, forKey: .passMS) ?? 0
    }
}

struct StreamSession: Sendable {
    let id = UUID()
    let speech: SpeechSession
    let options: AudioOptions
}

/// Wraps an error after which the engine's stream is ended, so the stream records the stop.
struct StreamFailure: Error {
    let error: any Error

    init(_ error: any Error) {
        self.error = error
    }
}

/// A live transcription started by `Whistle.stream`. Feed about one second of
/// PCM per `process` call and finish with `stop`, which commits the tail. The
/// engine holds one stream per process: while a stream is open, speech calls
/// other than its own `process` and `stop` throw `NeedleError.streamActive`.
/// A failed `process` ends the stream. Closing the Whistle also ends it.
public actor TranscriptionStream {
    static let stoppedError = NeedleError.invalidInput("stream is stopped")

    let session: StreamSession
    private let runtime: NativeRuntime
    private var stopped = false

    init(session: StreamSession, runtime: NativeRuntime) {
        self.session = session
        self.runtime = runtime
    }

    /// Appends PCM to the stream and returns the words it committed.
    public func process(_ pcm: [Float]) async throws -> StreamTranscript {
        try Task.checkCancellation()
        guard !stopped else { throw Self.stoppedError }
        try session.options.validate(pcm)
        do {
            return try await runtime.streamProcess(session, pcm: pcm)
        } catch let failure as StreamFailure {
            stopped = true
            throw failure.error
        }
    }

    /// Ends the stream and returns the words committed from its tail. A stopped
    /// stream cannot be resumed; start another with `Whistle.stream`.
    public func stop() async throws -> StreamTranscript {
        try Task.checkCancellation()
        guard !stopped else { throw Self.stoppedError }
        do {
            let tail = try await runtime.streamStop(session)
            stopped = true
            return tail
        } catch let failure as StreamFailure {
            stopped = true
            throw failure.error
        }
    }
}
