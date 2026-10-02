# Speech

`Whistle` loads the checksum-pinned ~16.9 MB `whistle.cact` archive from
[Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle) and uses
the Needle 3.1 engine. Speech weights download only when requested; standalone
transcription does not require text weights.

## Transcription

```swift
import Needle

let speech = try await Whistle()
let pcm: [Float] = [] // Replace with your 16 kHz mono samples.
let transcript = try await speech.transcribe(
    pcm,
    options: AudioOptions(language: "en", keywords: ["Paris", "Ada"], wordTimestamps: true)
)
print(transcript.text)
```

Supply mono `[Float]` PCM at 16000 Hz, with finite samples in `[-1, 1]`, up to
30 seconds. Empty clips and silence return empty text and language. `language`
is empty for detection, or `en`, `de`, `fr`, `es`, `it`, `nl`, or `pl`. Transcripts
include `ttftMS`, `decodeTPS`, and optional words with start/end times in seconds
and probabilities. File decoding, microphone capture, and resampling belong to
the caller.

## Audio tool calling

To transcribe and produce tool calls in one native turn, pass a Needle 3 agent
configured with your tools, such as the one in [Quick Start](../README.md#quick-start):

```swift
let response = try await speech.complete(pcm, agent: agent)
try response.validate()
print(response.audioText ?? "")
```

The response includes optional `audioText`, `audioLanguage`, `audioWords`,
`audioTTFTMS`, and `audioDecodeTPS` fields. This is a raw completion: it does not
execute handlers. Call `validate()` before acting on calls, and never execute
`suppressedCalls` automatically. Stateless agents reset once before the audio
request; stateful agents retain context for manual tool-result continuations
through `agent.complete`.

## Runtime and custom weights

Text and speech share the same serialized Needle 3 runtime and library path.
Loading or switching speech models preserves the active text conversation;
switching text agents follows the [conversation rules](../README.md#model-generations).
Audio options are cleared after every audio completion. Task cancellation cannot
interrupt native inference already in progress.

`WhistleConfiguration` accepts `weightsPath`, `libraryPath`, `cacheDirectory`,
and `bufferSize`. `weightsPath` selects a trusted custom speech archive; library
and cache selection follow the [engine configuration](../README.md#engine-downloads-and-caching).
The shared `.cact` header alone does not distinguish speech from text. Both
loading paths reject wrong-kind or unknown-kind archives before loading them.
The current speech classifier accepts the published version-1 audio manifest;
a future format requires a library update.

## Offline setup

`try await Engine.fetchWhistle(cacheDirectory: cacheDirectory)` prepares the
engine and speech weights without fetching text weights. Its `platform`
parameter selects the target macOS architecture. Preserve the `.sha256` markers
and use the same cache directory in `WhistleConfiguration`. For audio tool
calling, also prepare text weights with `Engine.fetch`.
