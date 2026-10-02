import Foundation

/// Reads only recognized v3 kind/head metadata; the native loader validates model tensors.
struct WeightMetadata {
    enum Kind { case text, speech }

    private let data: Data
    private let directory: Int
    private let count: Int
    private let manifest: Int
    private let width: Int
    private let layers: Int

    init?(_ data: Data) {
        guard data.count >= 196, data.integer(0, UInt32.self) == 0x05E1_2A84 else { return nil }
        let field = { Int(data.integer($0 * 4, UInt32.self)) }
        let directory = 196 + field(2) * 4
        let count = field(1)
        let manifest = 13 + field(10) * (field(19) == 0 ? 24 : 27) + field(31) * 4
        guard
            directory <= data.count, count <= (data.count - directory) / 44,
            field(7) > 0, field(10) > 0, field(31) <= 16, manifest < count else { return nil }
        self.data = data
        self.directory = directory
        self.count = count
        self.manifest = manifest
        width = field(7)
        layers = field(10)
    }

    var kind: Kind? {
        guard let tensor = tensor(manifest) else { return nil }
        if tensor.dtype == 4, manifest + 1 == count {
            return .text
        }
        if headCodes != nil {
            return .text
        }
        // ponytail: recognize the published v1 speech manifest; extend when its format changes.
        if tensor.matches(2, [13]), manifest + 1 < count {
            let value = { Float(bitPattern: data.integer(tensor.offset + $0 * 4, UInt32.self)) }
            if value(0) == 3246, value(1) == 1, value(9) == 16000 {
                return .speech
            }
        }
        return nil
    }

    private var headCodes: [UInt16]? {
        guard
            let tensor = tensor(manifest), tensor.dtype == 1, tensor.shape.count == 1,
            (1 ... 3).contains(tensor.shape[0]) else { return nil }
        let codes = (0 ..< tensor.shape[0]).map { data.integer(tensor.offset + $0 * 2, UInt16.self) }
        guard
            codes.allSatisfy({ [0x3C00, 0x4000, 0x4200].contains($0) }),
            zip(codes, codes.dropFirst()).allSatisfy({ $0 < $1 }) else { return nil }
        return codes
    }

    var hasConfidenceHead: Bool {
        guard
            let codes = headCodes, codes.contains(0x4000),
            let norm = tensor(manifest - 1), norm.matches(1, [width]),
            let manifestTensor = tensor(manifest, after: norm.end) else { return false }
        var index = manifest + 1
        var end = manifestTensor.end
        for code in codes {
            var head: [Tensor] = []
            for _ in 0 ..< 6 {
                guard let next = tensor(index, after: end) else { return false }
                head.append(next)
                index += 1
                end = next.end
            }
            guard validHead(head, code: code) else { return false }
            if code == 0x4200 {
                guard let calibration = tensor(index, after: end), calibration.matches(1, [3]) else { return false }
                index += 1
                end = calibration.end
            }
        }
        return tensor(index, after: end)?.dtype == 4 && index + 1 == count
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
        return (head[0].matches(3, [probeRows, width]) || head[0].matches(1, [layers + 1, probes, width]))
            && head[1].matches(1, [layers + 1, probes]) && head[2].matches(matrixType, [queries, width])
            && head[3].matches(1, [queries, layers + 1, probes])
            && head[4].matches(matrixType, [output, queryWidth]) && head[5].matches(1, [output])
            && (code != 0x4000 || output == 1) && (code != 0x4200 || output == 3)
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
        let record = directory + index * 44
        let dtype = data[record]
        let rank = Int(data[record + 1])
        guard rank <= 4, data.integer(record + 2, UInt16.self) == 0 else { return nil }
        let dimensions = (0 ..< 4).map { Int(data.integer(record + 4 + $0 * 4, UInt32.self)) }
        let shape = Array(dimensions.prefix(rank))
        guard
            shape.allSatisfy({ $0 > 0 }), dimensions.dropFirst(rank).allSatisfy({ $0 == 0 }),
            let offset = Int(exactly: data.integer(record + 20, UInt64.self)),
            let size = Int(exactly: data.integer(record + 28, UInt64.self)),
            offset >= max(directory + count * 44, previousEnd), offset <= data.count,
            size > 0, size <= data.count - offset else { return nil }
        let group = Int(data.integer(record + 36, UInt32.self))
        let bits = data.integer(record + 40, UInt32.self)
        guard Self.validSize(dtype: dtype, shape: shape, size: size, group: group, bits: bits) else { return nil }

        return Tensor(dtype: dtype, shape: shape, offset: offset, end: offset + size)
    }

    private static func validSize(dtype: UInt8, shape: [Int], size: Int, group: Int, bits: UInt32) -> Bool {
        switch dtype {
        case 1, 2: // FP16 or FP32.
            var bytes = dtype == 1 ? 2 : 4
            for dimension in shape {
                guard dimension <= size / bytes else { return false }
                bytes *= dimension
            }
            guard size == bytes, group == 0, bits == 0 else { return false }
        case 3: // CQ4 probe matrices, with padded input groups.
            guard shape.count == 2, bits == 4, group >= 2, group & (group - 1) == 0 else { return false }
            let bytesPerRow = ((shape[1] + group - 1) / group) * (group / 2 + 2)
            guard shape[0] <= size / bytesPerRow, size == shape[0] * bytesPerRow else { return false }
        case 4: // RAW tokenizer.
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
