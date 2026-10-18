import Foundation

/// The .cact archive layout written by upstream needle/model/export.py: a header
/// of 48 little-endian UInt32 fields and one Float32 (`_HDR_FMT "<48If"`), a
/// Float32 codebook, a directory of 44-byte tensor records (`_REC_FMT
/// "<BBHIIIIQQII"`) and then the tensor data.
enum Cact {
    static let needle2Tag: UInt32 = 0x05E1_2A83
    static let needle3Tag: UInt32 = 0x05E1_2A84

    static let headerSize = 49 * 4
    static let recordSize = 44
    static let maxRank = 4

    /// Header field indexes.
    enum Field {
        static let tensorCount = 1 // num_tensors
        static let codebookSize = 2 // Float32 codebook entries following the header
        static let modelWidth = 7 // d_model
        static let layers = 10 // num_layers
        static let qkvConvTaps = 19 // qkv_conv_taps; nonzero adds three tap tensors per layer
        static let engramSites = 31 // number of engram layers, at most maxEngramSites
        static let maxEngramSites = 16
    }

    /// Tensor record byte offsets.
    enum Record {
        static let dtype = 0 // UInt8
        static let rank = 1 // UInt8
        static let padding = 2 // UInt16, always zero
        static let shape = 4 // four UInt32 dimensions, zero beyond rank
        static let offset = 20 // UInt64 byte offset of the tensor data
        static let size = 28 // UInt64 byte length of the tensor data
        static let group = 36 // UInt32 CQ group size
        static let bits = 40 // UInt32 CQ index width
    }

    /// Tensor dtypes.
    enum DType {
        static let fp16: UInt8 = 1
        static let fp32: UInt8 = 2
        static let cq: UInt8 = 3
        static let raw: UInt8 = 4
    }

    /// Tensors before the optional head manifest or the speech manifest:
    /// embedding, the per-layer blocks, mHC, permutations, the per-site engrams
    /// and final_norm. Per-layer and per-site tensors are counted separately
    /// from the fixed ones.
    static let fixedTensors = 13
    static let layerTensors = 24
    static let layerTapTensors = 3
    static let engramTensors = 4

    /// heads.manifest codes, stored as the FP16 values 1, 2 and 3. Each head is
    /// [probes, gain, query, row_bias, proj, bias]; the router appends a
    /// three-value calibration vector.
    enum Head {
        static let embedding: UInt16 = 0x3C00
        static let confidence: UInt16 = 0x4000
        static let router: UInt16 = 0x4200
        static let codes = [embedding, confidence, router]
        static let maxCount = 3
        static let tensorCount = 6
        static let calibrationLength = 3
    }

    /// audio.manifest holds 13 Float32 values: a magic number, a layout version
    /// and the speech geometry, with the sample rate at index 9.
    enum AudioManifest {
        static let length = 13
        static let magic: Float = 3246
        static let version: Float = 1
        static let sampleRateIndex = 9
    }
}

/// Reads only recognized v3 kind/head metadata; the native loader validates model tensors.
struct WeightMetadata {
    enum Kind { case text, speech }

    private let data: Data
    private let directory: Int
    private let count: Int
    /// Index of the tensor record after final_norm: the head manifest of a text
    /// archive or the audio manifest of a speech archive.
    private let manifest: Int
    private let width: Int
    private let layers: Int

    init?(_ data: Data) {
        guard data.count >= Cact.headerSize, data.integer(0, UInt32.self) == Cact.needle3Tag else { return nil }
        let field = { Int(data.integer($0 * 4, UInt32.self)) }
        let directory = Cact.headerSize + field(Cact.Field.codebookSize) * 4
        let count = field(Cact.Field.tensorCount)
        let layers = field(Cact.Field.layers)
        let sites = field(Cact.Field.engramSites)
        let perLayer = Cact.layerTensors + (field(Cact.Field.qkvConvTaps) == 0 ? 0 : Cact.layerTapTensors)
        let manifest = Cact.fixedTensors + layers * perLayer + sites * Cact.engramTensors
        guard
            directory <= data.count, count <= (data.count - directory) / Cact.recordSize,
            field(Cact.Field.modelWidth) > 0, layers > 0, sites <= Cact.Field.maxEngramSites,
            manifest < count else { return nil }
        self.data = data
        self.directory = directory
        self.count = count
        self.manifest = manifest
        width = field(Cact.Field.modelWidth)
        self.layers = layers
    }

    var kind: Kind? {
        guard let tensor = tensor(manifest) else { return nil }
        if tensor.dtype == Cact.DType.raw, manifest + 1 == count {
            return .text // Text archive without probe heads: only the tokenizer follows.
        }
        if headCodes != nil {
            return .text
        }
        // Recognize the published version-1 audio manifest only; extend this
        // classifier when upstream publishes another speech archive layout.
        if tensor.matches(Cact.DType.fp32, [Cact.AudioManifest.length]), manifest + 1 < count {
            let value = { Float(bitPattern: data.integer(tensor.offset + $0 * 4, UInt32.self)) }
            let geometry = (value(0), value(1), value(Cact.AudioManifest.sampleRateIndex))
            if geometry == (Cact.AudioManifest.magic, Cact.AudioManifest.version, Float(Whistle.sampleRate)) {
                return .speech
            }
        }
        return nil
    }

    /// Decodes an FP16 heads.manifest. Codes must be strictly increasing and
    /// drawn from the known head kinds.
    private var headCodes: [UInt16]? {
        guard
            let tensor = tensor(manifest), tensor.dtype == Cact.DType.fp16, tensor.shape.count == 1,
            (1 ... Cact.Head.maxCount).contains(tensor.shape[0]) else { return nil }
        let codes = (0 ..< tensor.shape[0]).map { data.integer(tensor.offset + $0 * 2, UInt16.self) }
        guard
            codes.allSatisfy({ Cact.Head.codes.contains($0) }),
            zip(codes, codes.dropFirst()).allSatisfy({ $0 < $1 }) else { return nil }
        return codes
    }

    var hasConfidenceHead: Bool {
        guard
            let codes = headCodes, codes.contains(Cact.Head.confidence),
            let norm = tensor(manifest - 1), norm.matches(Cact.DType.fp16, [width]),
            let manifestTensor = tensor(manifest, after: norm.end) else { return false }
        var index = manifest + 1
        var end = manifestTensor.end
        for code in codes {
            var head: [Tensor] = []
            for _ in 0 ..< Cact.Head.tensorCount {
                guard let next = tensor(index, after: end) else { return false }
                head.append(next)
                index += 1
                end = next.end
            }
            guard validHead(head, code: code) else { return false }
            if code == Cact.Head.router {
                guard
                    let calibration = tensor(index, after: end),
                    calibration.matches(Cact.DType.fp16, [Cact.Head.calibrationLength]) else { return false }
                index += 1
                end = calibration.end
            }
        }
        return tensor(index, after: end)?.dtype == Cact.DType.raw && index + 1 == count
    }

    private func validHead(_ head: [Tensor], code: UInt16) -> Bool {
        guard head[1].shape.count == 2, head[2].shape.count == 2, head[5].shape.count == 1 else { return false }
        let probes = head[1].shape[1]
        let queries = head[2].shape[0]
        let output = head[5].shape[0]
        let (probeRows, probeOverflow) = (layers + 1).multipliedReportingOverflow(by: probes)
        let (queryWidth, queryOverflow) = queries.multipliedReportingOverflow(by: width)
        guard !probeOverflow, !queryOverflow else { return false }
        let matrixType = head[0].dtype
        // Probes are CQ ((L+1)*k, d_model), or FP16 (L+1, k, d_model) in the
        // earlier all-FP16 layout.
        return (head[0].matches(Cact.DType.cq, [probeRows, width])
            || head[0].matches(Cact.DType.fp16, [layers + 1, probes, width]))
            && head[1].matches(Cact.DType.fp16, [layers + 1, probes]) && head[2].matches(matrixType, [queries, width])
            && head[3].matches(Cact.DType.fp16, [queries, layers + 1, probes])
            && head[4].matches(matrixType, [output, queryWidth]) && head[5].matches(Cact.DType.fp16, [output])
            && (code != Cact.Head.confidence || output == 1) && (code != Cact.Head.router || output == 3)
    }

    private struct Tensor {
        let dtype: UInt8
        let shape: [Int]
        let offset: Int
        let end: Int

        func matches(_ dtype: UInt8, _ shape: [Int]) -> Bool {
            self.dtype == dtype && self.shape == shape
        }
    }

    private func tensor(_ index: Int, after previousEnd: Int = 0) -> Tensor? {
        guard (0 ..< count).contains(index) else { return nil }
        let record = directory + index * Cact.recordSize
        let dtype = data[record + Cact.Record.dtype]
        let rank = Int(data[record + Cact.Record.rank])
        guard rank <= Cact.maxRank, data.integer(record + Cact.Record.padding, UInt16.self) == 0 else { return nil }
        let dimensions = (0 ..< Cact.maxRank)
            .map { Int(data.integer(record + Cact.Record.shape + $0 * 4, UInt32.self)) }
        let shape = Array(dimensions.prefix(rank))
        guard
            shape.allSatisfy({ $0 > 0 }), dimensions.dropFirst(rank).allSatisfy({ $0 == 0 }),
            let offset = Int(exactly: data.integer(record + Cact.Record.offset, UInt64.self)),
            let size = Int(exactly: data.integer(record + Cact.Record.size, UInt64.self)),
            offset >= max(directory + count * Cact.recordSize, previousEnd), offset <= data.count,
            size > 0, size <= data.count - offset else { return nil }
        let group = Int(data.integer(record + Cact.Record.group, UInt32.self))
        let bits = data.integer(record + Cact.Record.bits, UInt32.self)
        guard Self.validSize(dtype: dtype, shape: shape, size: size, group: group, bits: bits) else { return nil }

        return Tensor(dtype: dtype, shape: shape, offset: offset, end: offset + size)
    }

    private static func validSize(dtype: UInt8, shape: [Int], size: Int, group: Int, bits: UInt32) -> Bool {
        switch dtype {
        case Cact.DType.fp16, Cact.DType.fp32:
            var bytes = dtype == Cact.DType.fp16 ? 2 : 4
            for dimension in shape {
                guard dimension <= size / bytes else { return false }
                bytes *= dimension
            }
            guard size == bytes, group == 0, bits == 0 else { return false }
        case Cact.DType.cq: // Current probe matrices are CQ4, with padded input groups.
            guard shape.count == 2, bits == 4, group >= 2, group & (group - 1) == 0 else { return false }
            let bytesPerRow = ((shape[1] + group - 1) / group) * (group / 2 + 2)
            guard shape[0] <= size / bytesPerRow, size == shape[0] * bytesPerRow else { return false }
        case Cact.DType.raw: // Tokenizer.
            guard shape.isEmpty, group == 0, bits == 0 else { return false }
        default: return false
        }
        return true
    }
}

private extension Data {
    /// Callers have checked the containing header, tensor record or payload bounds.
    func integer<T: FixedWidthInteger>(_ offset: Int, _: T.Type) -> T {
        withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
    }
}
