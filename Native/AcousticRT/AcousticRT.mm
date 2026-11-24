// AcousticRT: the path tracer's scene queries on Metal's ray-tracing hardware (README part 52; MethodTransport, PathWave).
// Unity's kernels write query records into a buffer; one Metal kernel answers them with Metal's intersector against an
// acceleration structure of the scene, on Unity's own command buffer (IUnityGraphicsMetalV2), so there is no CPU round
// trip. Only the intersection is here; the walk stays in Unity's HLSL.
//   Scene: one primitive acceleration structure per object in its own frame (built once), under an instance acceleration
//   structure whose transforms and masks a kernel writes from Unity's State buffer (the poses the HLSL walk reads) and which
//   is rebuilt before each update's walk (event Build), so moving and switched-off objects cost no CPU work.
//   Query record (3 x uint4): origin, tmin; direction, tmax; flags (bit 0: any hit), object + 1 (0: the whole scene; else
//   that object's mesh alone, in its pose: point location's probe), unused x 2.
//   Answer (uint4): the primitive's global index (0xffffffff: none), the distance's bits, 0, 0 (the walk computes the normal
//   from the primitive).
#import <Metal/Metal.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsMetal.h"

static const char *kSource = R"MSL(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
struct State { float4 position, rotation, velocity, angularVelocity; uint4 flags; };
struct ObjectAS { primitive_acceleration_structure as; };
struct Instance { packed_float3 columns[4]; uint options, mask, functionOffset, structureIndex; };
float3 rotateBy(float4 q, float3 v) { return v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
kernel void artInstances(device Instance *instances [[buffer(0)]], device const State *states [[buffer(1)]],
                         device const uint *instanceObject [[buffer(2)]], constant uint2 &params [[buffer(3)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= params.x) return;
    uint object = instanceObject[i];
    State s = states[params.y + object];
    Instance d;
    d.columns[0] = rotateBy(s.rotation, float3(1, 0, 0)); d.columns[1] = rotateBy(s.rotation, float3(0, 1, 0));
    d.columns[2] = rotateBy(s.rotation, float3(0, 0, 1)); d.columns[3] = s.position.xyz;
    d.options = 4u; // opaque
    d.mask = (s.flags.x & 3u) == 3u ? 1u : 0u; // active with geometry, as the walk's (flags & 3) == 3
    d.functionOffset = 0; d.structureIndex = i;
    instances[i] = d;
}
kernel void artIntersect(instance_acceleration_structure scene [[buffer(0)]], device const uint4 *queries [[buffer(1)]],
                         device uint4 *answers [[buffer(2)]], device const uint *count [[buffer(3)]],
                         device const uint *instanceObject [[buffer(4)]], device const uint *mapOffset [[buffer(5)]],
                         device const uint *primitiveMap [[buffer(6)]], device const State *states [[buffer(7)]],
                         device const ObjectAS *objects [[buffer(8)]], constant uint4 &params [[buffer(9)]],
                         uint i [[thread_position_in_grid]])
{
    if (i == 0 && params.z != 0) answers[0].zw = uint2(count[params.x], 0xC0FFEEu); // (diagnostic)
    if (i >= count[params.x]) return;
    uint4 q0 = queries[3 * i], q1 = queries[3 * i + 1], q2 = queries[3 * i + 2];
    ray r;
    r.origin = as_type<float3>(q0.xyz); r.min_distance = as_type<float>(q0.w);
    r.direction = as_type<float3>(q1.xyz); r.max_distance = as_type<float>(q1.w);
    bool anyHit = (q2.x & 1u) != 0;
    uint hit = 0xffffffffu; float t = r.max_distance;
    if (q2.y == 0)
    {
        intersector<triangle_data, instancing> x;
        x.accept_any_intersection(anyHit);
        auto h = x.intersect(r, scene, 1u);
        if (h.type == intersection_type::triangle) { hit = primitiveMap[mapOffset[instanceObject[h.instance_id]] + h.primitive_id]; t = h.distance; }
    }
    else
    {
        uint object = q2.y - 1;
        State s = states[params.y + object];
        float4 inverse = float4(-s.rotation.xyz, s.rotation.w);
        ray local = r;
        local.origin = rotateBy(inverse, r.origin - s.position.xyz); local.direction = rotateBy(inverse, r.direction);
        intersector<triangle_data> x;
        x.accept_any_intersection(anyHit);
        auto h = x.intersect(local, objects[object].as);
        if (h.type == intersection_type::triangle) { hit = primitiveMap[mapOffset[object] + h.primitive_id]; t = h.distance; }
    }
    answers[i] = uint4(hit, as_type<uint>(t), 0, 0);
}
)MSL";

static IUnityInterfaces *gInterfaces = nullptr;
static IUnityGraphicsMetalV2 *gMetal = nullptr;
static std::mutex gLock;
static std::string gError;

struct Scene
{
    id<MTLDevice> device;
    std::vector<id<MTLAccelerationStructure>> blas;     // per object with triangles (nil otherwise)
    std::vector<uint32_t> instanceObject;               // instance -> object
    id<MTLBuffer> instanceObjectBuffer, mapOffsetBuffer, primitiveMapBuffer, objectStructures, instances;
    id<MTLAccelerationStructure> tlas;
    id<MTLBuffer> tlasScratch;
    MTLInstanceAccelerationStructureDescriptor *tlasDescriptor;
    id<MTLComputePipelineState> instancePipeline, intersectPipeline;
    NSArray<id<MTLAccelerationStructure>> *structures;  // for useResources
    // Unity's buffers (registered once per solver)
    id<MTLBuffer> queries, answers, counts, args, states;
    uint32_t countIndex = 0, argsOffset = 0, queryOffset = 0, statesOffset = 0;
    bool built = false;
};
static std::map<int, Scene *> gScenes;
static int gNextHandle = 1;

static void Fail(const std::string &message) { std::lock_guard<std::mutex> g(gLock); gError = message; }

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces *interfaces)
{
    gInterfaces = interfaces;
    gMetal = interfaces->Get<IUnityGraphicsMetalV2>();
}
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload() { gMetal = nullptr; }

static id<MTLDevice> Device()
{
    if (gMetal == nullptr || gInterfaces == nullptr) return nil;
    IUnityGraphics *graphics = gInterfaces->Get<IUnityGraphics>();
    if (graphics == nullptr || graphics->GetRenderer() != kUnityGfxRendererMetal) return nil;
    return gMetal->MetalDevice();
}

// 1: hardware queries are available (Unity on Metal, a device with ray tracing); 0 otherwise.
extern "C" int UNITY_INTERFACE_EXPORT ARTSupported()
{
    id<MTLDevice> device = Device();
    return device != nil && device.supportsRaytracing ? 1 : 0;
}

extern "C" int UNITY_INTERFACE_EXPORT ARTLastError(char *buffer, int capacity)
{
    std::lock_guard<std::mutex> g(gLock);
    int n = (int)std::min<size_t>(gError.size(), capacity > 0 ? capacity - 1 : 0);
    if (capacity > 0) { memcpy(buffer, gError.data(), n); buffer[n] = 0; }
    gError.clear(); // (each error is reported once)
    return n;
}

// The scene: triangles in their objects' own frames (9 floats each) with their object, built into one primitive acceleration
// structure per object (synchronously, on a queue of its own). Returns a handle (> 0), or 0 (ARTLastError says why).
extern "C" int UNITY_INTERFACE_EXPORT ARTCreate(const float *vertices, const int *triangleObject, int triangleCount, int objectCount)
{
    @autoreleasepool
    {
        id<MTLDevice> device = Device();
        if (device == nil || !device.supportsRaytracing) { Fail("no Metal device with ray tracing"); return 0; }
        auto *scene = new Scene();
        scene->device = device;
        std::vector<std::vector<uint32_t>> owned(objectCount);
        for (int t = 0; t < triangleCount; t++)
            if (triangleObject[t] >= 0 && triangleObject[t] < objectCount) owned[triangleObject[t]].push_back((uint32_t)t);
        std::vector<uint32_t> mapOffset(std::max(objectCount, 1)), primitiveMap;
        scene->blas.assign(objectCount, nil);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> encoder = [commands accelerationStructureCommandEncoder];
        NSMutableArray<id<MTLAccelerationStructure>> *structures = [NSMutableArray array];
        std::vector<id<MTLBuffer>> keep;
        for (int o = 0; o < objectCount; o++)
        {
            mapOffset[o] = (uint32_t)primitiveMap.size();
            if (owned[o].empty()) continue;
            std::vector<float> local(9 * owned[o].size());
            for (size_t k = 0; k < owned[o].size(); k++)
            {
                memcpy(&local[9 * k], vertices + 9 * (size_t)owned[o][k], 9 * sizeof(float));
                primitiveMap.push_back(owned[o][k]);
            }
            id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:local.data() length:local.size() * sizeof(float) options:MTLResourceStorageModeShared];
            keep.push_back(vertexBuffer);
            auto *geometry = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            geometry.vertexBuffer = vertexBuffer; geometry.vertexStride = 12; geometry.triangleCount = owned[o].size(); geometry.opaque = YES;
            auto *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            descriptor.geometryDescriptors = @[geometry];
            MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:descriptor];
            id<MTLAccelerationStructure> structure = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            id<MTLBuffer> scratch = [device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 16) options:MTLResourceStorageModePrivate];
            keep.push_back(scratch);
            [encoder buildAccelerationStructure:structure descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
            scene->blas[o] = structure;
            scene->instanceObject.push_back((uint32_t)o);
            [structures addObject:structure];
        }
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) { Fail("acceleration structure build failed"); delete scene; return 0; }
        scene->structures = structures;
        if (primitiveMap.empty()) primitiveMap.push_back(0);
        std::vector<uint32_t> instanceObject = scene->instanceObject;
        if (instanceObject.empty()) instanceObject.push_back(0);
        scene->instanceObjectBuffer = [device newBufferWithBytes:instanceObject.data() length:instanceObject.size() * 4 options:MTLResourceStorageModeShared];
        scene->mapOffsetBuffer = [device newBufferWithBytes:mapOffset.data() length:mapOffset.size() * 4 options:MTLResourceStorageModeShared];
        scene->primitiveMapBuffer = [device newBufferWithBytes:primitiveMap.data() length:primitiveMap.size() * 4 options:MTLResourceStorageModeShared];
        // Each object's structure for the single-object queries (an argument buffer of resource IDs).
        std::vector<MTLResourceID> ids(std::max(objectCount, 1));
        for (int o = 0; o < objectCount; o++) if (scene->blas[o] != nil) ids[o] = scene->blas[o].gpuResourceID;
        scene->objectStructures = [device newBufferWithBytes:ids.data() length:ids.size() * sizeof(MTLResourceID) options:MTLResourceStorageModeShared];
        size_t instanceCount = scene->instanceObject.size();
        scene->instances = [device newBufferWithLength:std::max<size_t>(instanceCount, 1) * sizeof(MTLAccelerationStructureInstanceDescriptor) options:MTLResourceStorageModePrivate];
        auto *tlas = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlas.instancedAccelerationStructures = structures;
        tlas.instanceCount = instanceCount;
        tlas.instanceDescriptorBuffer = scene->instances;
        tlas.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
        scene->tlasDescriptor = tlas;
        MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:tlas];
        scene->tlas = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        scene->tlasScratch = [device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 16) options:MTLResourceStorageModePrivate];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:nil error:&error];
        if (library == nil) { Fail(std::string("kernel compile: ") + error.localizedDescription.UTF8String); delete scene; return 0; }
        scene->instancePipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"artInstances"] error:&error];
        scene->intersectPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"artIntersect"] error:&error];
        if (scene->instancePipeline == nil || scene->intersectPipeline == nil) { Fail("pipeline creation failed"); delete scene; return 0; }
        std::lock_guard<std::mutex> g(gLock);
        int handle = gNextHandle++;
        gScenes[handle] = scene;
        return handle;
    }
}

// Unity's buffers (ComputeBuffer.GetNativeBufferPtr): the query records (from byte queryOffset), the answers, the counts (the
// query count at uint index countIndex), the indirect arguments for the intersection (3 uints at byte argsOffset), and the
// poses (States, from byte statesOffset: a copy a kernel of the walk writes, since a buffer Unity uploads to by SetData may be
// replaced behind its native pointer).
extern "C" int UNITY_INTERFACE_EXPORT ARTSetBuffers(int handle, void *queries, int queryOffset, void *answers, void *counts, int countIndex,
                                                    void *args, int argsOffset, void *states, int statesOffset)
{
    std::lock_guard<std::mutex> g(gLock);
    auto found = gScenes.find(handle);
    if (found == gScenes.end()) { gError = "unknown handle"; return 0; }
    Scene *s = found->second;
    s->queries = (__bridge id<MTLBuffer>)queries; s->queryOffset = (uint32_t)queryOffset; s->answers = (__bridge id<MTLBuffer>)answers;
    s->counts = (__bridge id<MTLBuffer>)counts; s->countIndex = (uint32_t)countIndex;
    s->args = (__bridge id<MTLBuffer>)args; s->argsOffset = (uint32_t)argsOffset;
    s->states = (__bridge id<MTLBuffer>)states; s->statesOffset = (uint32_t)statesOffset;
    return 1;
}

enum { EventBuild = 1, EventIntersect = 2, EventDestroy = 3 };

static Scene *Find(int handle)
{
    std::lock_guard<std::mutex> g(gLock);
    auto found = gScenes.find(handle);
    return found == gScenes.end() ? nullptr : found->second;
}

// A command buffer of our own on Unity's queue, between Unity's work before the event and after it. At an event Unity's current
// command buffer is committed already (or is committed here: CommitCurrentCommandBuffer), and Unity waits only for its own
// command buffers (a readback of a buffer we wrote returned before our work had run), so ours signals an event that Unity's next
// command buffer, fresh and without an encoder, waits for first: Unity's later work and its waits then come after ours.
static id<MTLEvent> gOrder;
static uint64_t gOrderValue = 0;
static id<MTLCommandBuffer> Commands()
{
    gMetal->EndCurrentCommandEncoder();
    gMetal->CommitCurrentCommandBuffer();
    id<MTLCommandQueue> queue = gMetal->CommandQueue();
    id<MTLCommandBuffer> commands = queue == nil ? nil : [queue commandBuffer];
    if (commands == nil) Fail("no command queue");
    if (gOrder == nil) gOrder = [gMetal->MetalDevice() newEvent];
    return commands;
}
static void Done(id<MTLCommandBuffer> commands)
{
    [commands encodeSignalEvent:gOrder value:++gOrderValue];
    [commands addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if (c.status == MTLCommandBufferStatusError) Fail(std::string("command buffer failed: ") + (c.error ? c.error.localizedDescription.UTF8String : "?"));
    }];
    [commands commit];
    id<MTLCommandBuffer> next = gMetal->CurrentCommandBuffer();
    if (next == nil || next.status != MTLCommandBufferStatusNotEnqueued) { Fail("Unity's next command buffer is not fresh"); return; }
    [next encodeWaitForEvent:gOrder value:gOrderValue];
}
// Render thread. data: handle | state offset << 12 (Build: the frame's first State).
static void UNITY_INTERFACE_API OnEvent(int eventId, void *data)
{
    @autoreleasepool
    {
        uintptr_t packed = (uintptr_t)data;
        int handle = (int)(packed & 0xfff);
        uint32_t stateOffset = (uint32_t)(packed >> 12);
        if (gMetal == nullptr) return;
        if (eventId == EventDestroy)
        {
            Scene *s = nullptr;
            { std::lock_guard<std::mutex> g(gLock); auto f = gScenes.find(handle); if (f != gScenes.end()) { s = f->second; gScenes.erase(f); } }
            if (s == nullptr) return;
            id<MTLCommandBuffer> commands = gMetal->CurrentCommandBuffer();
            if (commands != nil) [commands addCompletedHandler:^(id<MTLCommandBuffer>) { delete s; }]; // after the GPU's last use
            else delete s;
            return;
        }
        Scene *s = Find(handle);
        if (s == nullptr || s->states == nil) { Fail("event for an unknown or unbound scene"); return; }
        uint32_t instanceCount = (uint32_t)s->instanceObject.size();
        if (eventId == EventBuild && instanceCount == 0) return;
        id<MTLCommandBuffer> commands = Commands();
        if (commands == nil) return;
        if (eventId == EventBuild)
        {
            id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
            [compute setComputePipelineState:s->instancePipeline];
            [compute setBuffer:s->instances offset:0 atIndex:0];
            [compute setBuffer:s->states offset:s->statesOffset atIndex:1];
            [compute setBuffer:s->instanceObjectBuffer offset:0 atIndex:2];
            uint32_t params[2] = {instanceCount, stateOffset};
            [compute setBytes:params length:sizeof(params) atIndex:3];
            [compute dispatchThreads:MTLSizeMake(instanceCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [compute endEncoding];
            id<MTLAccelerationStructureCommandEncoder> build = [commands accelerationStructureCommandEncoder];
            [build buildAccelerationStructure:s->tlas descriptor:s->tlasDescriptor scratchBuffer:s->tlasScratch scratchBufferOffset:0];
            [build endEncoding];
            Done(commands);
            s->built = true;
            return;
        }
        if (eventId == EventIntersect)
        {
            if (!s->built || s->queries == nil || s->answers == nil || s->counts == nil || s->args == nil) { Fail("intersect before build or buffers"); [commands commit]; return; }
            id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoder];
            [compute setComputePipelineState:s->intersectPipeline];
            [compute setAccelerationStructure:s->tlas atBufferIndex:0];
            [compute setBuffer:s->queries offset:s->queryOffset atIndex:1];
            [compute setBuffer:s->answers offset:0 atIndex:2];
            [compute setBuffer:s->counts offset:0 atIndex:3];
            [compute setBuffer:s->instanceObjectBuffer offset:0 atIndex:4];
            [compute setBuffer:s->mapOffsetBuffer offset:0 atIndex:5];
            [compute setBuffer:s->primitiveMapBuffer offset:0 atIndex:6];
            [compute setBuffer:s->states offset:s->statesOffset atIndex:7];
            [compute setBuffer:s->objectStructures offset:0 atIndex:8];
            uint32_t params[4] = {s->countIndex, stateOffset, 1, 0};
            [compute setBytes:params length:sizeof(params) atIndex:9];
            [compute useResource:s->tlas usage:MTLResourceUsageRead];
            if (s->structures.count > 0)
            {
                std::vector<id<MTLResource>> all(s->structures.count);
                for (NSUInteger k = 0; k < s->structures.count; k++) all[k] = s->structures[k];
                [compute useResources:all.data() count:all.size() usage:MTLResourceUsageRead];
            }
            [compute dispatchThreadgroupsWithIndirectBuffer:s->args indirectBufferOffset:s->argsOffset threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [compute endEncoding];
            Done(commands);
        }
    }
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT ARTEventFunction() { return OnEvent; }
