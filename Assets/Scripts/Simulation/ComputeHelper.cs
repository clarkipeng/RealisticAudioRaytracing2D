using UnityEngine;
using System.Collections.Generic;

namespace Helpers
{
    // Buffer and dispatch helpers used by the legacy renderer.
    public static class ComputeHelper
    {
        public static void Dispatch(ComputeShader cs, int numIterationsX, int numIterationsY = 1, int numIterationsZ = 1, int kernelIndex = 0)
        {
            Vector3Int threadGroupSizes = GetThreadGroupSizes(cs, kernelIndex);
            int numGroupsX = Mathf.CeilToInt(numIterationsX / (float)threadGroupSizes.x);
            int numGroupsY = Mathf.CeilToInt(numIterationsY / (float)threadGroupSizes.y);
            int numGroupsZ = Mathf.CeilToInt(numIterationsZ / (float)threadGroupSizes.z);

            if (numGroupsX == 0) numGroupsX = 1;
            if (numGroupsY == 0) numGroupsY = 1;
            if (numGroupsZ == 0) numGroupsZ = 1;

            cs.Dispatch(kernelIndex, numGroupsX, numGroupsY, numGroupsZ);
        }

        public static int GetStride<T>()
        {
            return System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));
        }

        public static void CreateAppendBuffer<T>(ref ComputeBuffer buffer, int count)
        {
            int stride = GetStride<T>();
            bool createNewBuffer = buffer == null || !buffer.IsValid() || buffer.count != count || buffer.stride != stride;
            if (createNewBuffer)
            {
                Release(buffer);
                buffer = new ComputeBuffer(count, stride, ComputeBufferType.Append);
            }

            buffer.SetCounterValue(0);
        }

        public static bool CreateStructuredBuffer<T>(ref ComputeBuffer buffer, int count)
        {
            int stride = GetStride<T>();
            bool createNewBuffer = buffer == null || !buffer.IsValid() || buffer.count != count || buffer.stride != stride;
            if (createNewBuffer)
            {
                Release(buffer);
                buffer = new ComputeBuffer(count, stride);
                return true;
            }

            return false;
        }

        public static ComputeBuffer CreateStructuredBuffer<T>(T[] data)
        {
            var buffer = new ComputeBuffer(data.Length, GetStride<T>());
            buffer.SetData(data);
            return buffer;
        }

        public static void CreateStructuredBuffer<T>(ref ComputeBuffer buffer, List<T> data) where T : struct
        {
            int stride = GetStride<T>();
            bool createNewBuffer = buffer == null || !buffer.IsValid() || buffer.count != data.Count || buffer.stride != stride;
            if (createNewBuffer)
            {
                Release(buffer);
                buffer = new ComputeBuffer(data.Count, stride);
            }

            buffer.SetData(data);
        }

        public static void CreateStructuredBuffer<T>(ref ComputeBuffer buffer, T[] data)
        {
            CreateStructuredBuffer<T>(ref buffer, data.Length);
            buffer.SetData(data);
        }

        public static void Release(ComputeBuffer buffer)
        {
            buffer?.Release();
        }

        public static void Release(params ComputeBuffer[] buffers)
        {
            foreach (var buffer in buffers)
            {
                Release(buffer);
            }
        }

        public static Vector3Int GetThreadGroupSizes(ComputeShader compute, int kernelIndex = 0)
        {
            uint x, y, z;
            compute.GetKernelThreadGroupSizes(kernelIndex, out x, out y, out z);
            return new Vector3Int((int)x, (int)y, (int)z);
        }
    }
}
