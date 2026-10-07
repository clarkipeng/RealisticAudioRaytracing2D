using System;
using System.IO;
using UnityEngine;

public static class TakeComparison
{
    [Serializable] public sealed class ReferenceCheck
    {
        public int responseStepSamples=1;
        public double maximumRmsError,maximumAbsoluteError;
    }
    [Serializable] public sealed class Metrics
    {
        public long samples;
        public int channels=1;
        public double rmsError,maximumAbsoluteError,referenceRms,snrDb;
        public bool exactMatch,snrDefined;
    }
    [Serializable] public sealed class Report
    {
        public string experiment="direct-temporal-resolution-v1";
        public string referenceStatus="finer temporal sampling of the same direct model; not a converged reflection teacher";
        public string sceneHash,stateHash,eventHash,sourceHash;
        public string takeDirectory; // Relative to this comparison directory; optional for older reports.
        public string[] listenerIds,listenerNames;
        public TakeRender.Report low,high,reference;
        public ReferenceCheck referenceCheck;
        public Metrics[] referenceErrors;
        public bool referenceAccepted,eligibleForTemporalTraining;
        public string trainingScope="direct-model temporal reconstruction only; no indirect/ray-budget training reference";
        public double measuredSpeedup;
        public Metrics[] listeners;
        public bool complete;
    }
    public static Report Run(AcousticTake take,TakeRender.Settings low,TakeRender.Settings high,string output,ComputeShader shader=null,Action<float> progress=null,ReferenceCheck referenceCheck=null)
    {
        TakeRender.Validate(low); TakeRender.Validate(high);
        var shape=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(low)); shape.responseStepSamples=high.responseStepSamples;
        if (JsonUtility.ToJson(shape)!=JsonUtility.ToJson(high) || low.responseStepSamples<=high.responseStepSamples)
            throw new ArgumentException("Temporal comparison requires a larger low-budget response step; all other settings must match.");
        TakeRender.Settings referenceSettings=null;
        if (referenceCheck!=null)
        {
            if (referenceCheck.responseStepSamples<1 || referenceCheck.responseStepSamples>=high.responseStepSamples ||
                !NonnegativeFinite(referenceCheck.maximumRmsError) || !NonnegativeFinite(referenceCheck.maximumAbsoluteError))
                throw new ArgumentException("Reference needs a finer step and explicit finite nonnegative error tolerances.");
            referenceSettings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(high));
            referenceSettings.responseStepSamples=referenceCheck.responseStepSamples; TakeRender.Validate(referenceSettings);
        }
        if (Directory.Exists(output)) throw new IOException("Comparison output directory already exists.");
        Directory.CreateDirectory(output);
        var report=new Report {sourceHash=AcousticTake.SourceHash(take.Metadata),referenceCheck=referenceCheck,sceneHash=take.Metadata.sceneHash,stateHash=take.Metadata.stateHash,eventHash=take.Metadata.eventHash};
        if(low.transport==TakeRender.Transport.Raytraced)
        {report.experiment="spectral-temporal-resolution-v1";report.referenceStatus="same geometric spectral preview at two response steps; no validated training teacher";report.trainingScope="preview comparison only";}
        report.takeDirectory=Path.GetRelativePath(Path.GetFullPath(output),Path.GetFullPath(take.DirectoryPath));
        report.listenerIds=new string[take.Metadata.scene.listeners.Length]; report.listenerNames=new string[report.listenerIds.Length];
        for(int l=0;l<report.listenerIds.Length;l++)
        {var obj=take.Metadata.scene.objects[take.Metadata.scene.listeners[l]]; report.listenerIds[l]=obj.id; report.listenerNames[l]=obj.name;}
        string path=Path.Combine(output,"comparison.json"); File.WriteAllText(path,JsonUtility.ToJson(report,true));
        // Warm the backend/JIT before either measured solve. File preparation remains part of each wall time.
        var state=new AcousticTake.State[take.Metadata.scene.objects.Length]; take.Evaluate(0,state);
        var response=new TakeRender.DirectPath[take.Metadata.scene.sources.Length*take.Metadata.scene.listeners.Length];
        using (var warmup=new DirectSolver(take.Metadata.scene,take.Geometry,high,shader,sampleRate:take.Metadata.sampleRate)) warmup.Solve(state,response);
        report.low=TakeRender.RenderDirect(take,low,Path.Combine(output,"low"),p=>progress?.Invoke(p/(referenceCheck==null ? 2:3)),shader);
        report.high=TakeRender.RenderDirect(take,high,Path.Combine(output,"high"),p=>progress?.Invoke((1+p)/(referenceCheck==null ? 2:3)),shader);
        if (report.low.outputSamples!=report.high.outputSamples || report.low.sceneHash!=report.high.sceneHash ||
            report.low.takeStateHash!=report.high.takeStateHash || report.low.takeEventHash!=report.high.takeEventHash ||
            report.low.takeSourceHash!=report.high.takeSourceHash || report.low.takeSourceHash!=report.sourceHash)
            throw new InvalidDataException("Comparison inputs/output timelines do not align.");
        report.listeners=new Metrics[take.Metadata.scene.listeners.Length];
        for (int l=0; l<report.listeners.Length; l++) report.listeners[l]=CompareWave(
            Path.Combine(output,"low","listener-"+l+".wav"),Path.Combine(output,"high","listener-"+l+".wav"));
        report.measuredSpeedup=report.low.wallSeconds>0 ? report.high.wallSeconds/report.low.wallSeconds:0;
        if (referenceSettings!=null)
        {
            report.reference=TakeRender.RenderDirect(take,referenceSettings,Path.Combine(output,"reference"),p=>progress?.Invoke((2+p)/3),shader);
            report.referenceErrors=new Metrics[report.listeners.Length]; report.referenceAccepted=true;
            for (int l=0;l<report.listeners.Length;l++)
            {
                var error=report.referenceErrors[l]=CompareWave(Path.Combine(output,"high","listener-"+l+".wav"),Path.Combine(output,"reference","listener-"+l+".wav"));
                report.referenceAccepted &= error.rmsError<=referenceCheck.maximumRmsError && error.maximumAbsoluteError<=referenceCheck.maximumAbsoluteError;
            }
            report.eligibleForTemporalTraining=report.referenceAccepted && low.transport==TakeRender.Transport.Direct;
            report.referenceStatus=report.referenceAccepted ? "high output agrees with finer same-model output within the configured tolerances":"high output fails the configured finer-reference tolerances";
        }
        report.complete=true;
        File.WriteAllText(Path.Combine(output,"comparison.complete.json"),JsonUtility.ToJson(report,true));
        File.Replace(Path.Combine(output,"comparison.complete.json"),path,null);
        return report;
    }
    static bool NonnegativeFinite(double value) => value>=0 && !double.IsInfinity(value);
    public static Metrics CompareWave(string low,string high)
    {
        using (var a=new BinaryReader(File.OpenRead(low))) using (var b=new BinaryReader(File.OpenRead(high)))
        {
            if (a.BaseStream.Length!=b.BaseStream.Length || a.BaseStream.Length<44 || (a.BaseStream.Length-44)%4!=0)
                throw new InvalidDataException("Waveform shapes differ.");
            for (int i=0; i<44; i++) if (a.ReadByte()!=b.ReadByte()) throw new InvalidDataException("Waveform formats differ.");
            long values=(a.BaseStream.Length-44)/4;
            a.BaseStream.Position=22; int channels=a.ReadUInt16(); a.BaseStream.Position=44;
            if(channels<1 || values%channels!=0) throw new InvalidDataException("Invalid audio channel layout.");
            var result=new Metrics {samples=values/channels,channels=channels};
            double error=0,power=0;
            for (long i=0; i<values; i++)
            {
                double x=a.ReadSingle(),y=b.ReadSingle();
                if (double.IsNaN(x) || double.IsInfinity(x) || double.IsNaN(y) || double.IsInfinity(y)) throw new InvalidDataException("Non-finite rendered audio.");
                double d=x-y; error+=d*d; power+=y*y; result.maximumAbsoluteError=Math.Max(result.maximumAbsoluteError,Math.Abs(d));
            }
            result.rmsError=Math.Sqrt(error/Math.Max(1,values)); result.referenceRms=Math.Sqrt(power/Math.Max(1,values));
            result.exactMatch=error==0; result.snrDefined=error>0 && power>0;
            if (result.snrDefined) result.snrDb=10*Math.Log10(power/error);
            return result;
        }
    }
}
