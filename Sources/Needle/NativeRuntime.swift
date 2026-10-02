import Darwin
import Foundation

struct NativeAPI: Sendable {
    typealias Initialize = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafePointer<CChar>?
    ) -> Int32
    typealias Complete = @convention(c) (
        UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias CompleteWithAudio = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<Float>?, Int32, Int32, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias Reset = @convention(c) () -> Void
    typealias Load = @convention(c) (UnsafeRawPointer?, UInt64) -> Int32
    typealias Embed = @convention(c) (UnsafePointer<CChar>?, UnsafeMutablePointer<Float>?, Int32) -> Int32
    typealias EmbedWithAudio = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<Float>?, Int32, UnsafeMutablePointer<Float>?, Int32
    ) -> Int32

    let initialize: @Sendable (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>?) -> Int32
    let complete: @Sendable (String, Int32, UnsafeMutableBufferPointer<CChar>) -> Int32
    let reset: @Sendable () -> Void
    let load: @Sendable (UnsafeRawPointer, UInt64) -> Int32

    var lastError: (@Sendable () -> String?)?
    var embed: (@Sendable (String, UnsafeMutableBufferPointer<Float>) -> Int32)?

    static func open(_ path: String) throws -> Self {
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            throw NeedleError.native("load library: \(String(cString: dlerror()))")
        }
        do {
            // The engine owns process-global pointers; keep it mapped for process lifetime.
            return try bind { dlsym(handle, $0) }
        } catch {
            dlclose(handle)
            throw error
        }
    }

    static func bind(resolve: (String) -> UnsafeMutableRawPointer?) throws -> Self {
        func symbol<T>(_ name: String, as _: T.Type) throws -> T {
            guard let address = resolve(name) else { throw NeedleError.native("missing symbol \(name)") }
            return unsafeBitCast(address, to: T.self)
        }
        let capabilities = ["needle_models", "needle_transcribe", "needle_set_audio"]
        let modern = capabilities.contains { resolve($0) != nil }
        if modern {
            for name in capabilities + ["needle_last_error", "needle_embed"] where resolve(name) == nil {
                throw NeedleError.native("unsupported native ABI: missing \(name)")
            }
        }
        let initialize = try symbol("needle_init", as: Initialize.self)
        let complete: @Sendable (String, Int32, UnsafeMutableBufferPointer<CChar>) -> Int32
        if modern {
            let function = try symbol("needle_complete", as: CompleteWithAudio.self)
            complete = { text, tokens, buffer in
                text.withCString { function($0, nil, 0, tokens, buffer.baseAddress, Int32(buffer.count)) }
            }
        } else {
            let function = try symbol("needle_complete", as: Complete.self)
            complete = { text, tokens, buffer in
                text.withCString { function($0, tokens, buffer.baseAddress, Int32(buffer.count)) }
            }
        }
        let reset = try symbol("needle_reset", as: Reset.self)
        let load = try symbol("needle_load", as: Load.self)
        var api = Self(
            initialize: { initialize($0, $1, $2) },
            complete: complete,
            reset: { reset() },
            load: { load($0, $1) }
        )
        if let address = resolve("needle_last_error") {
            let function = unsafeBitCast(address, to: (@convention(c) () -> UnsafePointer<CChar>?).self)
            api.lastError = { function().map { String(cString: $0) } }
        }
        api.bindEmbedding(resolve("needle_embed"), modern: modern)
        return api
    }

    private mutating func bindEmbedding(_ address: UnsafeMutableRawPointer?, modern: Bool) {
        guard let address else { return }
        if modern {
            let function = unsafeBitCast(address, to: EmbedWithAudio.self)
            embed = { text, output in
                text.withCString { function($0, nil, 0, output.baseAddress, Int32(output.count)) }
            }
        } else {
            let function = unsafeBitCast(address, to: Embed.self)
            embed = { text, output in
                text.withCString { function($0, output.baseAddress, Int32(output.count)) }
            }
        }
    }

    /// Read the process-global error before any other native call can replace it.
    func failure(_ operation: String, code: Int32, fallback: String = "") -> NeedleError {
        let diagnostic = (lastError?() ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = diagnostic.isEmpty ? fallback.trimmingCharacters(in: .whitespacesAndNewlines) : diagnostic
        return .native("\(operation) failed with code \(code)" + (detail.isEmpty ? "" : ": \(detail)"))
    }
}

// ponytail: one actor per generation serializes its global C engine; independent same-generation engines need processes.
actor NativeRuntime {
    private static let needle2 = NativeRuntime()
    private static let needle3 = NativeRuntime()

    static func shared(for generation: Int) -> NativeRuntime {
        generation == 2 ? needle2 : needle3
    }

    private var api: NativeAPI?
    private var libraryPath: String?
    private var activeID: UUID?
    private var activeWeights: String?
    private var weightsBlob: NSData?
    private var calibratedWeights = false
    private var initializationStorage: [NSData] = []

    init(api: NativeAPI? = nil) {
        self.api = api
    }

    func initialize(_ session: Session, library: URL) throws {
        try Task.checkCancellation()
        let path = library.standardizedFileURL.path
        if let libraryPath, libraryPath != path {
            throw NeedleError.native("engine already loaded from \(libraryPath)")
        }
        if api == nil {
            api = try NativeAPI.open(path)
        }
        libraryPath = path
        try bind(session)
    }

    func complete(_ session: Session, text: String, tokens: Int32, reset: Bool = false) throws -> Response {
        try Task.checkCancellation()
        let api = try bind(session)
        if reset {
            api.reset()
        }
        var buffer = [CChar](repeating: 0, count: session.bufferSize)
        let code = buffer.withUnsafeMutableBufferPointer { api.complete(text, tokens, $0) }
        // Cancellation cannot interrupt the native ABI; observe it when inference returns.
        try Task.checkCancellation()
        let end = buffer.firstIndex(of: 0)
        let bytes = buffer.prefix(end ?? buffer.count).map { UInt8(bitPattern: $0) }
        if code < 0 {
            let detail = String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw api.failure("complete", code: code, fallback: detail)
        }
        guard end != nil else { throw NeedleError.native("engine response exceeds buffer") }
        var response = try JSONDecoder().decode(Response.self, from: Data(bytes))
        guard !response.type.isEmpty else { throw NeedleError.native("response type is empty") }
        if session.tuned, !calibratedWeights {
            response.confidence = nil
        }
        return response
    }

    func reset(_ session: Session) throws {
        try Task.checkCancellation()
        try bind(session).reset()
    }

    func embed(_ session: Session, text: String) throws -> [Float] {
        try Task.checkCancellation()
        guard session.generation == 3 else { throw NeedleError.invalidInput("embeddings require Needle 3") }
        try validateCString(text, name: "input")
        let api = try bind(session)
        guard let embed = api.embed else { throw NeedleError.native("engine does not support embeddings") }
        let size = embed(text, UnsafeMutableBufferPointer(start: nil, count: 0))
        guard size >= 0 else { throw api.failure("embed size", code: size) }
        // ponytail: cap vectors at 4 MiB; raise if a future model needs wider features.
        guard size > 0, size <= 1 << 20 else { throw NeedleError.native("invalid embedding size \(size)") }
        try Task.checkCancellation()
        var output = [Float](repeating: 0, count: Int(size))
        let code = output.withUnsafeMutableBufferPointer { embed(text, $0) }
        try Task.checkCancellation()
        guard code >= 0 else { throw api.failure("embed", code: code) }
        guard code == size else { throw NeedleError.native("embedding length \(code), want \(size)") }
        return output
    }

    @discardableResult
    private func bind(_ session: Session) throws -> NativeAPI {
        guard let api else { throw NeedleError.native("engine is not loaded") }
        if activeID == session.id {
            return api
        }
        if session.weightsPath == nil, activeWeights != nil {
            throw NeedleError.native("tuned weights cannot be unloaded; use a separate process for the base model")
        }
        if let path = session.weightsPath, path != activeWeights {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let generation = try Self.weightsGeneration(data)
            guard generation == session.generation else {
                throw NeedleError
                    .invalidInput("weights generation changed: got \(generation), want \(session.generation)")
            }
            let metadata = WeightMetadata(data)
            guard generation != 3 || metadata?.kind == .text else {
                throw NeedleError.invalidInput("weights are not a recognized text model")
            }
            let blob = NSData(data: data)
            let code = api.load(blob.bytes, UInt64(blob.length))
            guard code >= 0 else { throw api.failure("load weights", code: code) }
            // NSData supplies a stable allocation for engines that retain the weight pointer.
            weightsBlob = blob
            activeWeights = path
            calibratedWeights = generation == 3 && metadata?.hasConfidenceHead == true
        }
        activeID = nil
        let storage = ([session.system, session.tools] + [session.toolIndexPath].compactMap(\.self))
            .map { NSData(data: Data($0.utf8) + [0]) }
        // Keep C strings alive for engines that retain init arguments, including failed rebinds.
        initializationStorage += storage
        let code = api.initialize(
            storage[0].bytes.assumingMemoryBound(to: CChar.self),
            storage[1].bytes.assumingMemoryBound(to: CChar.self),
            storage.count == 3 ? storage[2].bytes.assumingMemoryBound(to: CChar.self) : nil
        )
        guard code >= 0 else { throw api.failure("initialize", code: code) }
        initializationStorage = storage
        activeID = session.id
        return api
    }

    static func weightsGeneration(_ data: Data) throws -> Int {
        guard data.count >= 4 else { throw NeedleError.invalidInput("weights header is truncated") }
        let tag = data.prefix(4).enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << ($1.offset * 8) }
        switch tag {
        case 0x05E1_2A83: return 2
        case 0x05E1_2A84: return 3
        default: throw NeedleError.invalidInput("unknown weights header 0x\(String(tag, radix: 16))")
        }
    }
}
