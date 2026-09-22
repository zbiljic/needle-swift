import Foundation

public struct Configuration: Sendable {
    public var tools: [Tool]
    public var system: String
    /// Selects 2 or 3; zero defaults to 3. Custom weights take precedence.
    public var generation: Int
    public var weightsPath: String?
    public var toolIndexPath: String?
    public var bufferSize: Int
    /// A trusted dylib matching the selected generation. Falls back to its environment override, then download.
    public var libraryPath: String?
    public var cacheDirectory: URL?

    public init(
        tools: [Tool] = [],
        system: String = "",
        generation: Int = 3,
        weightsPath: String? = nil,
        toolIndexPath: String? = nil,
        bufferSize: Int = 64 * 1024,
        libraryPath: String? = nil,
        cacheDirectory: URL? = nil
    ) {
        self.tools = tools
        self.system = system
        self.generation = generation
        self.weightsPath = weightsPath
        self.toolIndexPath = toolIndexPath
        self.bufferSize = bufferSize
        self.libraryPath = libraryPath
        self.cacheDirectory = cacheDirectory
    }
}

/// A conversation using the process-wide engine for its model generation.
public final class Agent: Sendable {
    public static let defaultMaxSteps = 8
    public static let defaultMaxNewTokens = 512

    let session: Session
    let runtime: NativeRuntime
    private let handlers: [String: Tool.Handler]

    public convenience init(configuration: Configuration = Configuration()) async throws {
        try await self.init(configuration: configuration, runtime: nil)
    }

    init(configuration: Configuration, runtime: NativeRuntime?) async throws {
        try Task.checkCancellation()
        var prepared = try Session(configuration)
        handlers = Dictionary(uniqueKeysWithValues: configuration.tools.compactMap { tool in
            tool.handler.map { (tool.schema.name, $0) }
        })
        let override = prepared.libraryOverride(configuration.libraryPath)
        let library: URL
        if let override, !override.isEmpty {
            try validateCString(override, name: "library path")
            library = URL(fileURLWithPath: override).standardizedFileURL
        } else {
            library = try await Engine.fetchLibrary(
                generation: prepared.generation, cacheDirectory: configuration.cacheDirectory
            )
        }
        if prepared.generation == 3, !prepared.tuned {
            prepared.weightsPath = try await Engine.fetchBaseWeights(cacheDirectory: configuration.cacheDirectory).path
        }
        session = prepared
        self.runtime = runtime ?? NativeRuntime.shared(for: prepared.generation)
        try await self.runtime.initialize(session, library: library)
    }

    /// Raw inference; does not apply engine validation warnings or execute tools.
    public func complete(_ text: String, maxNewTokens: Int = defaultMaxNewTokens) async throws -> Response {
        let tokens = maxNewTokens == 0 ? Self.defaultMaxNewTokens : maxNewTokens
        guard tokens > 0, tokens <= Int(Int32.max) else {
            throw NeedleError.invalidInput("invalid max new tokens \(maxNewTokens)")
        }
        try validateCString(text, name: "input")
        return try await runtime.complete(session, text: text, tokens: Int32(tokens))
    }

    /// Runs up to `maxSteps` tool rounds. A validation error carries the rejected
    /// response and results of earlier rounds in `NeedleError.validation`.
    public func run(
        _ query: String,
        maxSteps: Int = defaultMaxSteps,
        maxNewTokens: Int = defaultMaxNewTokens
    ) async throws -> Response {
        let steps = maxSteps == 0 ? Self.defaultMaxSteps : maxSteps
        guard steps > 0 else { throw NeedleError.invalidInput("invalid max steps \(maxSteps)") }
        var response = try await complete(query, maxNewTokens: maxNewTokens)
        var executed: [JSONValue] = []
        response.results = executed
        try response.validate()
        for _ in 0 ..< steps {
            guard response.type == "call", let calls = response.functionCalls, !calls.isEmpty else { break }
            var results: [JSONValue] = []
            for call in calls {
                try Task.checkCancellation()
                let result = try await execute(call)
                results.append(result)
                executed.append(result)
            }
            let payload = try JSONEncoder().encode(results)
            response = try await complete(String(decoding: payload, as: UTF8.self), maxNewTokens: maxNewTokens)
            response.results = executed
            try response.validate()
        }
        return response
    }

    public func reset() async throws {
        try await runtime.reset(session)
    }

    private func execute(_ call: FunctionCall) async throws -> JSONValue {
        guard let handler = handlers[call.name] else {
            return .object(["error": .string("unknown tool: \(call.name)")])
        }
        do {
            let arguments = call.arguments.flatMap { $0 == .null ? nil : $0 } ?? .object([:])
            let result = try await handler(arguments)
            try Task.checkCancellation()
            return result
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            return .object(["error": .string(error.localizedDescription)])
        }
    }
}

struct Session: Sendable {
    let id = UUID()
    let system: String
    let tools: String
    let generation: Int
    let tuned: Bool
    var weightsPath: String?
    let toolIndexPath: String?
    let bufferSize: Int

    init(_ config: Configuration) throws {
        var seen: Set<String> = []
        for tool in config.tools {
            guard !tool.schema.name.isEmpty, seen.insert(tool.schema.name).inserted else {
                throw NeedleError.invalidInput("empty or duplicate tool name: \(tool.schema.name)")
            }
        }
        guard (2 ... Int(Int32.max)).contains(config.bufferSize) else {
            throw NeedleError.invalidInput("invalid buffer size \(config.bufferSize)")
        }
        try validateCString(config.system, name: "system")
        try validateCString(config.toolIndexPath ?? "", name: "tool index path")
        try validateCString(config.weightsPath ?? "", name: "weights path")
        system = config.system
        tools = try String(decoding: JSONEncoder().encode(config.tools.map(\.schema)), as: UTF8.self)
        weightsPath = config.weightsPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0).standardized.path }
        tuned = weightsPath != nil
        generation = try weightsPath.map { try Engine.weightsGeneration(at: $0) }
            ?? EngineRelease(config.generation).generation
        toolIndexPath = config.toolIndexPath.flatMap { $0.isEmpty ? nil : $0 }
        bufferSize = config.bufferSize
    }

    func libraryOverride(
        _ path: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        [path, environment["NEEDLE\(generation)_LIB_PATH"], generation == 2 ? environment["NEEDLE_LIB_PATH"] : nil]
            .compactMap(\.self).first { !$0.isEmpty }
    }
}

func validateCString(_ value: String, name: String) throws {
    guard !value.utf8.contains(0) else { throw NeedleError.invalidInput("\(name) contains NUL") }
}
