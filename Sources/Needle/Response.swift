import Foundation

public enum NeedleError: Error, LocalizedError {
    case invalidInput(String)
    case native(String)
    case download(String)
    case validation(Response)

    public var errorDescription: String? {
        switch self {
        case let .invalidInput(message), let .native(message), let .download(message):
            "needle: \(message)"
        case let .validation(response):
            if response.validation?.negation == true {
                "needle: response validation: negation detected"
            } else {
                "needle: response validation: ungrounded fields: "
                    + (response.validation?.ungrounded ?? []).joined(separator: ", ")
            }
        }
    }
}

public struct FunctionCall: Codable, Sendable, Equatable {
    public var name: String
    public var arguments: JSONValue?

    public init(name: String, arguments: JSONValue? = nil) {
        self.name = name
        self.arguments = arguments
    }
}

public struct Validation: Codable, Sendable, Equatable {
    public var ungrounded: [String]?
    public var negation: Bool?

    public init(ungrounded: [String] = [], negation: Bool = false) {
        self.ungrounded = ungrounded
        self.negation = negation
    }
}

/// Raw engine output. Optional fields accommodate sparse native envelopes.
public struct Response: Codable, Sendable, Equatable {
    /// Known values are `call`, `respond`, `refuse`, and `text`.
    public var type: String
    public var success: Bool?
    public var error: String?
    public var errorCode: String?
    public var functionCalls: [FunctionCall]?
    /// Calls withheld by the engine. `run` never executes these automatically.
    public var suppressedCalls: [FunctionCall]?
    public var reasoning: String?
    public var confidence: Double?
    public var prefillTPS: Double?
    public var decodeTPS: Double?
    public var peakRAMMB: Double?
    public var results: [JSONValue]?
    public var validation: Validation?

    enum CodingKeys: String, CodingKey {
        case type, success, error, reasoning, confidence, results, validation
        case errorCode = "error_code"
        case functionCalls = "function_calls"
        case suppressedCalls = "suppressed_calls"
        case prefillTPS = "prefill_tps"
        case decodeTPS = "decode_tps"
        case peakRAMMB = "peak_ram_mb"
    }

    public init(type: String, functionCalls: [FunctionCall] = [], validation: Validation? = nil) {
        self.type = type
        self.functionCalls = functionCalls
        self.validation = validation
    }

    /// Rejects the entire response if the engine flags negation or ungrounded fields.
    /// Unflagged output still requires application-specific validation.
    public func validate() throws {
        if validation?.negation == true || !(validation?.ungrounded ?? []).isEmpty {
            throw NeedleError.validation(self)
        }
    }

    /// Decodes exactly one call's arguments. Call `validate()` before acting on them.
    public func extract<Value: Decodable>(_ type: Value.Type = Value.self) throws -> Value {
        guard self.type == "call", let calls = functionCalls, calls.count == 1 else {
            throw NeedleError.invalidInput("extract requires exactly one function call")
        }
        return try (calls[0].arguments ?? .null).decode(type)
    }
}
