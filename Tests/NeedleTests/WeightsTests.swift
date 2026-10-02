import Foundation
@testable import Needle
import Testing

// swiftlint:disable:next cyclomatic_complexity
func testWeights(heads: [UInt16] = [], quantized: Bool = true, speech: Bool = false) -> Data {
    // Metadata fixtures only; placeholder model tensors are never sent to a real engine.
    var tensors: [(UInt8, [Int])] = Array(repeating: (1, [1]), count: 66)
    tensors.append((1, [4]))
    if speech {
        tensors.append((2, [13]))
    } else if !heads.isEmpty {
        tensors.append((1, [heads.count]))
    }
    for code in heads {
        let output = code == 0x4000 ? 1 : (code == 0x4200 ? 3 : 2)
        let matrix: UInt8 = quantized ? 3 : 1
        tensors += [
            (matrix, quantized ? [3, 4] : [3, 1, 4]), (1, [3, 1]), (matrix, [2, 4]),
            (1, [2, 3, 1]), (matrix, [output, 8]), (1, [output]),
        ]
        if code == 0x4200 {
            tensors.append((1, [3]))
        }
    }
    tensors.append((4, []))
    let directory = 196 + 28 * 4
    var data = Data(repeating: 0, count: directory + tensors.count * 44)
    for (field, value) in [0: 0x05E1_2A84, 1: tensors.count, 2: 28, 7: 4, 10: 2, 19: 3] {
        data.put(UInt32(value), at: field * 4)
    }
    for (index, tensor) in tensors.enumerated() {
        let (dtype, shape) = tensor
        let offset = data.count
        var size = shape.reduce(dtype == 2 ? 4 : 2, *)
        if dtype == 3 {
            size = shape[0] * ((shape[1] + 3) / 4) * 4
        }
        if dtype == 4 {
            size = 3
        }
        data.append(Data(repeating: 0, count: size))
        let record = directory + index * 44
        data[record] = dtype
        data[record + 1] = UInt8(shape.count)
        for (dimension, length) in shape.enumerated() {
            data.put(UInt32(length), at: record + 4 + dimension * 4)
        }
        data.put(UInt64(offset), at: record + 20)
        data.put(UInt64(size), at: record + 28)
        if dtype == 3 {
            data.put(UInt32(4), at: record + 36)
            data.put(UInt32(4), at: record + 40)
        }
        if index == 67 {
            for (index, code) in heads.enumerated() {
                data.put(code, at: offset + index * 2)
            }
            if speech {
                for (index, value): (Int, Float) in [(0, 3246), (1, 1), (9, 16000)] {
                    data.put(value.bitPattern, at: offset + index * 4)
                }
            }
        }
    }
    return data
}

private extension Data {
    mutating func put(_ value: some FixedWidthInteger, at offset: Int) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { replaceSubrange(offset ..< offset + $0.count, with: $0) }
    }
}

@Test
func recognizesWeightMetadataAndRejectsMalformedConfidence() {
    for quantized in [false, true] {
        for heads: [UInt16] in [[], [0x3C00], [0x4000], [0x4200], [0x4000, 0x4200], [0x3C00, 0x4000, 0x4200]] {
            let metadata = WeightMetadata(testWeights(heads: heads, quantized: quantized))
            #expect(metadata?.kind == .text)
            #expect(metadata?.hasConfidenceHead == heads.contains(0x4000))
        }
    }
    #expect(WeightMetadata(testWeights(speech: true))?.kind == .speech)
    for heads: [UInt16] in [[0x4000, 0x4000], [0x4000, 0x3C00], [0x4000, 0x7E00]] {
        #expect(WeightMetadata(testWeights(heads: heads))?.kind == nil)
    }
    let valid = testWeights(heads: [0x4000, 0x4200])
    for size in 0 ..< valid.count {
        #expect(WeightMetadata(Data(valid.prefix(size)))?.hasConfidenceHead != true)
    }
    let manifest = 308 + 67 * 44
    for offset in [
        0,
        4,
        8,
        28,
        40,
        124,
        manifest,
        manifest + 4,
        manifest + 20,
        manifest + 28,
        manifest + 44 + 4,
        manifest + 44 + 36,
        manifest + 44 + 40,
    ] {
        var invalid = valid
        invalid.put(UInt32.max, at: offset)
        #expect(WeightMetadata(invalid)?.hasConfidenceHead != true)
    }
    var overflow = valid
    overflow.put(UInt64.max, at: manifest + 20)
    #expect(WeightMetadata(overflow)?.kind == nil)
    overflow = valid
    overflow.put(UInt64.max, at: manifest + 28)
    #expect(WeightMetadata(overflow)?.kind == nil)
}

@Test(arguments: [false, true])
func confidenceUsesLoadedBytesAndRejectsWrongKind(calibrated: Bool) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("custom.cact")
    try testWeights(heads: calibrated ? [0x4000] : []).write(to: path)
    let fake = FakeNative(Array(repeating: #"{"type":"call","confidence":0.9}"#, count: 4))
    let runtime = NativeRuntime(api: fake.api)
    let library = URL(fileURLWithPath: "/test/libneedle.dylib")
    let first = try Session(Configuration(weightsPath: path.path))
    try await runtime.initialize(first, library: library)
    try testWeights(heads: calibrated ? [] : [0x4000]).write(to: path)
    let second = try Session(Configuration(weightsPath: path.path))
    for session in [first, second] {
        #expect(try await (runtime.complete(session, text: "hello", tokens: 1).confidence != nil) == calibrated)
    }
    #expect(fake.snapshot.loads == 1)
    let otherPath = directory.appendingPathComponent("other.cact")
    try testWeights().write(to: otherPath)
    try await runtime.initialize(Session(Configuration(weightsPath: otherPath.path)), library: library)
    #expect(try await (runtime.complete(first, text: "hello", tokens: 1).confidence != nil) == !calibrated)
    let badPath = directory.appendingPathComponent("speech.cact")
    for data in [testWeights(speech: true), Data([0x84, 0x2A, 0xE1, 0x05])] {
        try data.write(to: badPath)
        let wrong = try Session(Configuration(weightsPath: badPath.path))
        await #expect(throws: NeedleError.self) { try await runtime.initialize(wrong, library: library) }
    }
    #expect(fake.snapshot.loads == 3)
}
