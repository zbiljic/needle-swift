import Darwin
import Foundation

struct NativeAPI: Sendable {
    typealias Initialize = @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafePointer<CChar>?
    ) -> Int32
    typealias Complete = @convention(c) (
        UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<CChar>?, Int32
    ) -> Int32
    typealias Reset = @convention(c) () -> Void
    typealias Load = @convention(c) (UnsafeRawPointer?, UInt64) -> Int32

    let initialize: @Sendable (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>?) -> Int32
    let complete: @Sendable (String, Int32, UnsafeMutableBufferPointer<CChar>) -> Int32
    let reset: @Sendable () -> Void
    let load: @Sendable (UnsafeRawPointer, UInt64) -> Int32

    static func open(_ path: String) throws -> Self {
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            throw NeedleError.native("load library: \(String(cString: dlerror()))")
        }
        func symbol<T>(_ name: String, as _: T.Type) throws -> T {
            guard let address = dlsym(handle, name) else {
                throw NeedleError.native("missing symbol \(name)")
            }
            return unsafeBitCast(address, to: T.self)
        }
        do {
            let initialize = try symbol("needle_init", as: Initialize.self)
            let complete = try symbol("needle_complete", as: Complete.self)
            let reset = try symbol("needle_reset", as: Reset.self)
            let load = try symbol("needle_load", as: Load.self)
            // The engine owns process-global pointers; keep its library mapped for process lifetime.
            return Self(
                initialize: { initialize($0, $1, $2) },
                complete: { text, tokens, buffer in
                    text.withCString { complete($0, tokens, buffer.baseAddress, Int32(buffer.count)) }
                },
                reset: { reset() },
                load: { load($0, $1) }
            )
        } catch {
            dlclose(handle)
            throw error
        }
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

    func complete(_ session: Session, text: String, tokens: Int32) throws -> Response {
        try Task.checkCancellation()
        let api = try bind(session)
        var buffer = [CChar](repeating: 0, count: session.bufferSize)
        let code = buffer.withUnsafeMutableBufferPointer { api.complete(text, tokens, $0) }
        // Cancellation cannot interrupt the native ABI; observe it when inference returns.
        try Task.checkCancellation()
        let end = buffer.firstIndex(of: 0)
        let bytes = buffer.prefix(end ?? buffer.count).map { UInt8(bitPattern: $0) }
        if code < 0 {
            let detail = String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw NeedleError.native("complete failed with code \(code)" + (detail.isEmpty ? "" : ": \(detail)"))
        }
        guard end != nil else { throw NeedleError.native("engine response exceeds buffer") }
        var response = try JSONDecoder().decode(Response.self, from: Data(bytes))
        guard !response.type.isEmpty else { throw NeedleError.native("response type is empty") }
        if session.tuned {
            response.confidence = nil
        }
        return response
    }

    func reset(_ session: Session) throws {
        try Task.checkCancellation()
        try bind(session).reset()
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
            let blob = NSData(data: data)
            let code = api.load(blob.bytes, UInt64(blob.length))
            guard code >= 0 else { throw NeedleError.native("load weights failed with code \(code)") }
            // NSData supplies a stable allocation for engines that retain the weight pointer.
            weightsBlob = blob
            activeWeights = path
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
        guard code >= 0 else { throw NeedleError.native("initialize failed with code \(code)") }
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
