using NUnit.Framework;
using System;
using UnityEngine;

public partial class TakeChecks
{

    // Settings saved before the hybrid, pressure and energy methods and the impulse-response synthesis were removed
    // (method 0-2, synthesis 0) load as the path method with spectral synthesis.
    [TestCase("{\"transport\":1}")]
    [TestCase("{\"method\":0,\"synthesis\":0}")]
    [TestCase("{\"method\":2,\"synthesis\":1}")]
    [TestCase("{\"method\":3}"), Category("Core")]
    public void SavedSettingsLoadAsThePathMethod(string json)
    {
        var settings=JsonUtility.FromJson<TakeRender.Settings>(json);
        Assert.That(settings.method,Is.EqualTo(TakeRender.Method.Path));
        Assert.That(settings.synthesis,Is.EqualTo(TakeRender.Synthesis.Spectral));
        Assert.DoesNotThrow(()=>TakeRender.Validate(settings));
    }

    [Test, Category("Core")]
    public void DopplerRejectsAliasing()
    {
        // Pitch-up must attenuate content that would fold above output Nyquist in sinc mode.
        var tone=new float[128];for(int n=0;n<tone.Length;n++)tone[n]=Mathf.Sin(2*Mathf.PI*.375f*n);
        var signal=new TakeRender.Signal(tone);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        double energy=0;for(int n=0;n<256;n++){float v=Doppler.Read(signal,500.25+1.5*n,1.5,TakeRender.DelayFilter.WindowedSinc,32);energy+=v*v;}
        Check(Math.Sqrt(energy/256)<.01,"Doppler sinc low-pass rejects a tone shifted above Nyquist.");
    }

    [TestCase(-3),TestCase(3),Category("Core")]
    public void DefaultResamplingPreservesMovingTone(double receiverSpeed)
    {
        const int rate=44100,frequency=4000;
        var tone=new float[rate];
        for(int n=0;n<tone.Length;n++)tone[n]=.25f*(float)Math.Sin(2*Math.PI*frequency*n/rate);
        var signal=new TakeRender.Signal(tone);signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var settings=new TakeRender.Settings();double ratio=1-receiverSpeed/343,error=0,energy=0;
        // A single constant-velocity arrival has an exact sinusoidal source clock.
        // Vary the fractional read position; stationary/integer reads miss resampling flutter.
        for(int n=0;n<8192;n++)
        {
            double emission=1000.25+n*ratio,expected=.25*Math.Sin(2*Math.PI*frequency*emission/rate);
            double actual=Doppler.Read(signal,emission,ratio,settings.delayFilter,settings.sincRadius);
            error+=(actual-expected)*(actual-expected);energy+=expected*expected;
        }
        Assert.That(Math.Sqrt(error/energy),Is.LessThan(.001),"Default resampling must preserve the analytic tone within 0.1% RMS.");
    }
}
