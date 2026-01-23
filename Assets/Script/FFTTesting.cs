using UnityEngine;

public class FFTTesting : MonoBehaviour
{
    public ComputeShader raytraceShader;

    void Start()
    {
        TestFFT();
    }

    void TestFFT()
    {
        int N = 128;
        
        // 1. Use Vector2 to match 'float2' in HLSL (Real, Imaginary)

        int Hz = 440;
        int sampleRate = 48000;
        float totalTime = N / (float)sampleRate;
        float cycles = Hz * totalTime;
        float period = N / cycles;

        Vector2[] data = new Vector2[N];
        
        // Initialize input: Real = Sine Wave, Imaginary = 0
        for (int i = 0; i < N; i++)
        {
            float val = Mathf.Sin(2 * Mathf.PI * i / period);
            data[i] = new Vector2(val, 0f);
        }

        int k_fft = raytraceShader.FindKernel("FFT");
        int k_ifft = raytraceShader.FindKernel("IFFT");

        // 2. Create ONE buffer (Stride is automatically handled by Vector2 size)
        ComputeBuffer buffer = new ComputeBuffer(N, sizeof(float) * 2);
        buffer.SetData(data);

        // --- PERFORM FFT ---
        raytraceShader.SetBuffer(k_fft, "Data", buffer);
        raytraceShader.Dispatch(k_fft, 1, 1, 1); // 1 group of 128 threads

        // Read back FFT results (Optional, for debugging)
        Vector2[] fftOutput = new Vector2[N];
        buffer.GetData(fftOutput);

        // Print FFT
        Debug.Log($"FFT Output: [{string.Join(", ", fftOutput)}]");
        
        // Log a few bins (Bin 5 should be the peak)
        Debug.Log($"FFT Bin 5 Magnitude: {fftOutput[5]}"); 
        Debug.Log($"FFT Bin 123 Magnitude: {fftOutput[123]}"); 

        // --- PERFORM IFFT ---
        // The buffer currently holds frequency data; we transform it back in-place
        raytraceShader.SetBuffer(k_ifft, "Data", buffer);
        raytraceShader.Dispatch(k_ifft, 1, 1, 1);

        // Read back final result
        Vector2[] finalOutput = new Vector2[N];
        buffer.GetData(finalOutput);

        // --- VERIFY ---
        float totalError = 0f;
        for (int i = 0; i < N; i++)
        {
            // 3. Compare with original input
            // Note: No division by N here (the shader did it)
            float original = Mathf.Sin(2 * Mathf.PI * 5 * i / N);
            float sign = (finalOutput[i].x < 0) ? -1f : 1f;
            float reconstructed = sign * Mathf.Sqrt(finalOutput[i].x * finalOutput[i].x + finalOutput[i].y * finalOutput[i].y);
            // Debug.Log($"Index {i}: Original = {original:F4}, Reconstructed = {reconstructed:F4}"); 
            
            totalError += Mathf.Abs(original - reconstructed);
        }

        Debug.Log($"Total Absolute Error: {totalError:F6}");
        
        // Cleanup
        buffer.Release();
    }
}