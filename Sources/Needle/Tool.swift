import Foundation

public struct ToolSchema: Codable, Sendable, Equatable {
    public var name: String
    public var description: String
    public var parameters: [String: JSONValue]

    public init(
        name: String,
        description: String = "",
        parameters: [String: JSONValue] = ["type": .string("object"), "properties": .object([:])]
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// A schema with an optional Swift handler. Schema-only tools work with `complete`.
public struct Tool: Sendable {
    public typealias Handler = @Sendable (JSONValue) async throws -> JSONValue

    public let schema: ToolSchema
    public let handler: Handler?

    public init(schema: ToolSchema, handler: Handler? = nil) {
        self.schema = schema
        self.handler = handler
    }

    /// Uses an explicit JSON schema and Codable conversion, without reflection or macros.
    public init<Arguments: Decodable & Sendable>(
        schema: ToolSchema,
        handler: @escaping @Sendable (Arguments) async throws -> some Encodable & Sendable
    ) {
        self.schema = schema
        self.handler = { arguments in
            let decoded = try arguments.decode(Arguments.self)
            return try await JSONValue(handler(decoded))
        }
    }
}
