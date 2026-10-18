import Foundation

/// Errors thrown by this library. Pattern-match the cases to react to a
/// condition; `localizedDescription` carries the detail.
public enum NeedleError: Error, LocalizedError, Sendable {
    /// A caller-supplied value was rejected before reaching the engine.
    case invalidInput(String)
    /// A generation other than 2 or 3, or a feature the selected generation lacks.
    case unsupportedGeneration(String)
    /// A native library with an incomplete capability set.
    case unsupportedABI(missing: String)
    /// The model or engine has no embeddings.
    case embeddingUnsupported(String)
    /// The engine has no speech ABI.
    case speechUnsupported(String)
    /// An engine call returned a failure code. `detail` carries the engine's
    /// diagnostic when one is available.
    case nativeFailure(operation: String, code: Int32, detail: String)
    /// Engine or library state reported by this library rather than by a native code.
    case native(String)
    /// A live transcription is open on the engine; other speech calls wait for its `stop`.
    case streamActive
    /// The `Agent` or `Whistle` was closed.
    case closed
    /// An engine or weights download could not be completed or verified.
    case download(String)
    /// The engine flagged negation or ungrounded fields. `run` throws this before
    /// executing any tool call of the flagged turn.
    case validation(Response)

    public var errorDescription: String? {
        switch self {
        case let .invalidInput(message), let .native(message), let .download(message),
             let .unsupportedGeneration(message), let .embeddingUnsupported(message),
             let .speechUnsupported(message):
            "needle: \(message)"
        case let .unsupportedABI(missing):
            "needle: unsupported native ABI: missing \(missing)"
        case .streamActive:
            "needle: a live transcription is in progress"
        case .closed:
            "needle: closed"
        case let .nativeFailure(operation, code, detail):
            "needle: \(operation) failed with code \(code)" + (detail.isEmpty ? "" : ": \(detail)")
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
