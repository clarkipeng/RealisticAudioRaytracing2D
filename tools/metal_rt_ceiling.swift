// The hardware ray-tracing ceiling on this Mac (docs/benchmarks/path-denoise-2026-11-12/README.md, part 46): Metal's
// intersector against an acceleration structure of a scene dumped by MeasurePathLiveBudget (ACOUSTIC_PATH_BUDGET_DUMP:
// world-space triangles, then rays as position and direction), the same three queries as the tracer's own ceiling kernel
// (MethodPath.hlsl PathTraceBench): a closest hit in a fresh uniform direction, a closest hit along the ray's own direction,
// and an any-hit query between two rays' positions (a shadow ray). Queries per second, median of 7 dispatches of 2^20.
//     swiftc -O tools/metal_rt_ceiling.swift -o .context/path-100k/metal_rt_ceiling
//     python3 tools/heavy_guard.py run -- .context/path-100k/metal_rt_ceiling SCENE.bin
import Foundation
import Metal

let source = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
struct Ray6 { float px, py, pz, dx, dy, dz; };
uint hashRay(uint x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
float random01(thread uint &x) { x = hashRay(x + 0x9e3779b9); return (x >> 8) * (1.0 / 16777216.0); }
kernel void bench(primitive_acceleration_structure scene [[buffer(0)]], device const Ray6 *rays [[buffer(1)]],
                  constant uint &count [[buffer(2)]], constant uint &mode [[buffer(3)]], device uint *sink [[buffer(4)]],
                  uint index [[thread_position_in_grid]])
{
    Ray6 r = rays[index % count];
    uint rng = hashRay(index * 0x9e3779b9u + 0x2545f491u);
    ray q; q.origin = float3(r.px, r.py, r.pz); q.min_distance = 1e-4; q.max_distance = 1e4;
    float3 d = float3(r.dx, r.dy, r.dz);
    if (mode == 0) { float a = 2 * M_PI_F * random01(rng), z = 2 * random01(rng) - 1, s = sqrt(max(0.0, 1 - z * z)); d = float3(s * cos(a), s * sin(a), z); }
    if (mode == 2) { Ray6 t = rays[hashRay(index) % count]; d = float3(t.px, t.py, t.pz) - q.origin; float l = length(d); d /= max(l, 1e-6); q.max_distance = max(l - 1e-4, 0.0); }
    q.direction = d;
    intersector<triangle_data> i;
    i.accept_any_intersection(mode == 2);
    auto h = i.intersect(q, scene);
    if (h.distance < 0) sink[0] = h.primitive_id;
}
"""

func fail(_ message: String) -> Never { FileHandle.standardError.write((message + "\n").data(using: .utf8)!); exit(1) }
guard CommandLine.arguments.count > 1, let data = FileManager.default.contents(atPath: CommandLine.arguments[1]) else { fail("usage: metal_rt_ceiling SCENE.bin") }
var offset = 0
func readInt() -> Int { let v = data.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.load(as: Int32.self) }; offset += 4; return Int(v) }
func readFloats(_ n: Int) -> [Float] { let v = data.subdata(in: offset..<offset + 4 * n).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }; offset += 4 * n; return v }
let triangles = readInt(); let vertices = readFloats(9 * triangles)
let rayCount = readInt(); let rays = readFloats(6 * rayCount)
guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else { fail("no Metal ray tracing") }
let queue = device.makeCommandQueue()!
let vertexBuffer = device.makeBuffer(bytes: vertices, length: 4 * vertices.count, options: .storageModeShared)!
let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
geometry.vertexBuffer = vertexBuffer; geometry.vertexStride = 12; geometry.triangleCount = triangles
let descriptor = MTLPrimitiveAccelerationStructureDescriptor(); descriptor.geometryDescriptors = [geometry]
let sizes = device.accelerationStructureSizes(descriptor: descriptor)
let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)!
var build = queue.makeCommandBuffer()!
let encoder = build.makeAccelerationStructureCommandEncoder()!
encoder.build(accelerationStructure: structure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
encoder.endEncoding(); build.commit(); build.waitUntilCompleted()
let library = try! device.makeLibrary(source: source, options: nil)
let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "bench")!)
let rayBuffer = device.makeBuffer(bytes: rays, length: 4 * rays.count, options: .storageModeShared)!
let sink = device.makeBuffer(length: 4, options: .storageModeShared)!
let threads = 1 << 20
print("scene: \(triangles) triangles, \(rayCount) rays; acceleration structure \(sizes.accelerationStructureSize / 1024) KiB")
for (mode, name) in [(0, "closest hit, fresh directions"), (1, "closest hit, own directions"), (2, "any hit between rays")] {
    var rates: [Double] = []
    for _ in 0..<7 {
        let commands = queue.makeCommandBuffer()!
        let compute = commands.makeComputeCommandEncoder()!
        compute.setComputePipelineState(pipeline)
        compute.setAccelerationStructure(structure, bufferIndex: 0)
        compute.setBuffer(rayBuffer, offset: 0, index: 1)
        var count = UInt32(rayCount), m = UInt32(mode)
        compute.setBytes(&count, length: 4, index: 2); compute.setBytes(&m, length: 4, index: 3)
        compute.setBuffer(sink, offset: 0, index: 4)
        compute.useResource(structure, usage: .read)
        compute.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        compute.endEncoding(); commands.commit(); commands.waitUntilCompleted()
        rates.append(Double(threads) / (commands.gpuEndTime - commands.gpuStartTime))
    }
    rates.sort()
    print(String(format: "%@: %.0f M/s", name, rates[3] / 1e6))
}
