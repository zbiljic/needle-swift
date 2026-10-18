import Foundation

public struct AudioOptions: Sendable, Equatable {
    /// Empty detects the language; otherwise en, de, fr, es, it, nl or pl.
    public var language: String
    public var keywords: [String]
    public var wordTimestamps: Bool

    public init(language: String = "", keywords: [String] = [], wordTimestamps: Bool = false) {
        self.language = language
        self.keywords = keywords
        self.wordTimestamps = wordTimestamps
    }

    func validate(_ pcm: [Float]) throws {
        guard pcm.count <= Int(Int32.max) else {
            throw NeedleError.invalidInput("audio has \(pcm.count) samples; want at most \(Int32.max)")
        }
        guard pcm.allSatisfy({ $0.isFinite && (-1 ... 1).contains($0) }) else {
            throw NeedleError.invalidInput("PCM samples must be finite values in [-1, 1]")
        }
        guard ["", "en", "de", "fr", "es", "it", "nl", "pl"].contains(language) else {
            throw NeedleError.invalidInput("unsupported speech language \(language)")
        }
        try validateCString(keywords.joined(separator: "\n"), name: "speech keywords")
    }

    func withCStringOptions<T>(_ body: (UnsafePointer<CChar>?, UnsafePointer<CChar>?, Int32) -> T) -> T {
        let keywords = keywords.joined(separator: "\n")
        return language.withCString { languagePointer in
            keywords.withCString { keywordPointer in
                body(
                    language.isEmpty ? nil : languagePointer,
                    keywords.isEmpty ? nil : keywordPointer,
                    wordTimestamps ? 1 : 0
                )
            }
        }
    }
}

public struct WordTimestamp: Codable, Sendable, Equatable {
    public var word: String
    public var start: Double
    public var end: Double
    public var probability: Double
}

public struct Transcript: Codable, Sendable, Equatable {
    public var text: String
    public var language: String
    public var words: [WordTimestamp]?
    public var ttftMS: Double
    public var decodeTPS: Double

    enum CodingKeys: String, CodingKey {
        case text, language, words
        case ttftMS = "ttft_ms"
        case decodeTPS = "decode_tps"
    }
}

public struct WhistleConfiguration: Sendable {
    public var weightsPath: String?
    public var libraryPath: String?
    public var cacheDirectory: URL?
    public var bufferSize: Int

    public init(
        weightsPath: String? = nil,
        libraryPath: String? = nil,
        cacheDirectory: URL? = nil,
        bufferSize: Int = 256 * 1024
    ) {
        self.weightsPath = weightsPath
        self.libraryPath = libraryPath
        self.cacheDirectory = cacheDirectory
        self.bufferSize = bufferSize
    }
}

/// Opt-in speech using the same serialized process-wide engine as Needle 3 text.
public final class Whistle: Sendable {
    public static let sampleRate = 16000
    /// Bounds transcript JSON per second of audio, as upstream does, so long
    /// clips with word timestamps fit the output buffer.
    static let outputBytesPerSecond = 4096

    let session: SpeechSession
    let runtime: NativeRuntime

    public convenience init(configuration: WhistleConfiguration = WhistleConfiguration()) async throws {
        try await self.init(configuration: configuration, runtime: nil)
    }

    init(configuration: WhistleConfiguration, runtime: NativeRuntime?) async throws {
        try Task.checkCancellation()
        guard (2 ... Int(Int32.max)).contains(configuration.bufferSize) else {
            throw NeedleError.invalidInput("invalid buffer size \(configuration.bufferSize)")
        }
        try validateCString(configuration.weightsPath ?? "", name: "speech weights path")
        let library = try await Engine.resolveLibrary(
            generation: 3, path: configuration.libraryPath, cacheDirectory: configuration.cacheDirectory
        )
        let weights: URL = if let path = configuration.weightsPath, !path.isEmpty {
            URL(fileURLWithPath: path).standardizedFileURL
        } else {
            try await Engine.fetchSpeechWeights(cacheDirectory: configuration.cacheDirectory)
        }
        session = SpeechSession(weightsPath: weights.path, bufferSize: configuration.bufferSize)
        self.runtime = runtime ?? NativeRuntime.shared(for: 3)
        try await self.runtime.initializeSpeech(session, library: library)
    }

    /// Transcribes 16 kHz mono PCM of any length: up to 30 seconds runs as one pass, and
    /// longer audio runs as consecutive 30 second windows with word times counted from the
    /// start of the clip. Empty clips and silence produce empty text.
    public func transcribe(_ pcm: [Float], options: AudioOptions = AudioOptions()) async throws -> Transcript {
        try Task.checkCancellation()
        try options.validate(pcm)
        return try await runtime.transcribe(session, pcm: pcm, options: options)
    }

    /// Raw audio inference. Validate the response before acting on any returned tool calls.
    public func complete(
        _ pcm: [Float],
        agent: Agent,
        options: AudioOptions = AudioOptions(),
        maxNewTokens: Int = Agent.defaultMaxNewTokens
    ) async throws -> Response {
        try Task.checkCancellation()
        guard agent.session.generation == 3, agent.runtime === runtime else {
            throw NeedleError.invalidInput("audio completion requires a Needle 3 agent sharing the speech engine")
        }
        let tokens = try Agent.tokens(maxNewTokens)
        try options.validate(pcm)
        return try await runtime.completeSpeech(
            session,
            text: agent.session,
            pcm: pcm,
            options: options,
            tokens: tokens
        )
    }

    /// Prepares a live transcription with the language and keywords of `options`;
    /// `wordTimestamps` is ignored because every step carries word times. Nothing
    /// reaches the engine until the first `process`.
    public func stream(options: AudioOptions = AudioOptions()) throws -> TranscriptionStream {
        try options.validate([])
        return TranscriptionStream(session: StreamSession(speech: session, options: options), runtime: runtime)
    }

    /// Ends an open stream of this Whistle and releases its binding. Later calls
    /// throw `NeedleError.closed`. Loaded speech weights stay in the engine for
    /// other Whistle values.
    public func close() async {
        await runtime.closeSpeech(session)
    }
}

struct SpeechSession: Sendable {
    let id = UUID()
    let weightsPath: String
    /// Output capacity for clips up to about a minute; longer clips grow it.
    let bufferSize: Int

    func outputCapacity(samples: Int) -> Int {
        max(bufferSize, samples / Whistle.sampleRate * Whistle.outputBytesPerSecond)
    }
}
