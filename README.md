# needle-swift

Swift library for the Cactus [Needle 2](https://huggingface.co/Cactus-Compute/needle2)
and [Needle 3](https://huggingface.co/Cactus-Compute/needle3) on-device tool-calling
models.

It provides:

- native engine loading
- Needle 2 and Needle 3, with Needle 3 as the default
- automatic, checksum-verified engine and base-weight downloads
- high-level and manual completion loops
- typed Swift tool handlers
- structured response extraction

> This project is in early development and its API may change.

## Install

Requires **Swift 6.3+ and macOS 13+** (Apple Silicon or Intel). This initial version
supports macOS desktop processes. Linux, Windows, iOS, and sandboxed App Store
distribution are not supported.

Add `https://github.com/zbiljic/needle-swift` in Xcode using the `main` branch,
or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/zbiljic/needle-swift.git", branch: "main"),
],
targets: [
    .executableTarget(
        name: "MyApp",
        dependencies: [.product(name: "Needle", package: "needle-swift")]
    ),
]
```

## Quick Start

```swift
import Needle

struct WeatherArguments: Codable, Sendable {
    let city: String
}

let weather = Tool(
    schema: ToolSchema(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: [
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string")]),
            ]),
            "required": .array([.string("city")]),
        ]
    )
) { (arguments: WeatherArguments) in
    "Clear in \(arguments.city)"
}

let agent = try await Agent(configuration: Configuration(tools: [weather]))
let response = try await agent.run("What is the weather in Lagos?")
print(response.results ?? [])
```

Handlers are `@Sendable`, can be asynchronous, and decode arguments through
`Decodable` before execution. Their `Encodable` results return to the engine as
JSON. Supply schemas explicitly: Swift's `Codable` does not describe JSON Schema
constraints or field documentation.

Unknown tools and handler failures become `{"error":"..."}` tool results.
Cancellation propagates as an error. Validate arguments in the handler before
performing application actions; decoding does not enforce all JSON Schema rules.

## Manual completion and extraction

Using the weather schema from Quick Start, request arguments without executing
the handler:

```swift
let agent = try await Agent(configuration: Configuration(
    tools: [Tool(schema: weather.schema)]
))
let response = try await agent.complete("What is the weather in Lagos?")
try response.validate()
let arguments: WeatherArguments = try response.extract()
print(arguments.city)
try await agent.reset()
```

`complete` returns a `Response` without executing tools. `extract` decodes the
arguments of exactly one function call in a `call` response. Neither checks
engine validation warnings automatically; call `validate()` before acting on
the arguments.

`run` checks each response before executing its calls. If the engine flags
negation or ungrounded fields, it throws `NeedleError.validation(response)`.
The error includes results from earlier completed rounds; those actions are
not undone. Unflagged output still needs application validation.

The defaults are eight tool rounds and 512 new tokens per completion. Reaching
the round limit returns the latest response, which can still contain unexecuted
calls. A `refuse` response ends the loop without executing further tools.
Engine `success`, `error`, reasoning, performance statistics, and confidence
remain available on `Response`.
`suppressedCalls` preserves calls withheld by the engine; `run` never executes them.

Task cancellation is checked before and after inference and between tool calls.
It cannot interrupt native inference already in progress.

## Model generations

Needle **3** is the default (`generation: 0` also selects 3). Select Needle 2
to use its embedded base model:

```swift
let agent = try await Agent(configuration: Configuration(generation: 2))
```

`weightsPath` accepts Needle 2 or Needle 3 `.cact` files. The file header selects
the generation, overriding `Configuration.generation`. You can inspect it with
`Engine.weightsGeneration(at:)`; this checks the header, not the full archive.
Supplying custom weights sets `confidence` to `nil`, even for base weights.
Automatically downloaded Needle 3 base weights retain confidence.

Both generations can run in the same process, each with **one global
conversation**. Within a generation, switching agents resets the conversation;
switching back does not restore it. Do not overlap `run`, `complete`, or `reset`
operations within a generation, even on the same agent. Use separate processes
for independent concurrent conversations of the same generation.

After loading custom Needle 2 weights, returning to its embedded base model
requires a separate process. Needle 3 can reload its base archive.

## Engine downloads and caching

By default, the library downloads and caches the pinned Needle 3 engine (3.0.1)
and `needle3.cact` base weights (~35 MB). Subsequent runs use the cached files.

```swift
let library = try await Engine.fetch(generation: 3) // Library and base weights, ready for offline use.
let cached = try Engine.cached(generation: 3)       // Library lookup only; no download.
let version = try Engine.version(for: 2)            // "2.0.4"
```

The library selects an engine matching the resolved generation in this order:

1. `Configuration.libraryPath`.
2. `NEEDLE2_LIB_PATH` or `NEEDLE3_LIB_PATH`. The legacy `NEEDLE_LIB_PATH` is a
   fallback for Needle 2 only.
3. A checksum-verified download of the pinned Needle 2.0.4 or Needle 3.0.1 engine.

A Needle 3 library override still fetches missing base weights unless
`weightsPath` is supplied. Manual libraries must be trusted and ABI-compatible
with the selected generation. Only one library path per generation can be used
in a process.

Default caches are `~/.cache/cactus-needle/<engine-version>/<platform>/`. A custom
`cacheDirectory` is the exact destination directory. Needle 2 uses
`libneedle.dylib`; Needle 3 uses `libneedle3.dylib`, so both can share a custom
directory for one platform.

For offline use, call `Engine.fetch` for the target platform and generation,
preserve the downloaded files and `.sha256` markers, and use the same cache
directory at runtime. `Engine.cached` only locates the library; it does not
verify checksums or check for base weights.

## Development

Use Xcode's Swift 6.3+ toolchain and [mise](https://mise.jdx.dev/).

```sh
mise trust
mise run setup          # Install pinned tools
mise run                # List available tasks
mise run fmt            # Format Swift source
mise run check          # Formatting, lint, and offline tests
mise run test:native    # Downloads and tests both real engines
mise run clean          # Remove package build artifacts
swift build -c release
```

## License

Licensed under the [MIT License](LICENSE).

The downloaded engine and model retain their upstream license terms.
