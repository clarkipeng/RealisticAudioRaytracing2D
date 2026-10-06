using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

// Editor-only readers and audition. Runtime rendering and training do not depend on this code.
internal static class TakeComparisonAudio
{
    public const int MaximumPreviewFrames=2000000;
    public sealed class Wave : IDisposable
    {
        readonly BinaryReader reader;
        public readonly int Rate,Channels;
        public readonly long Frames;
        public Wave(string path,string hash,TakeRender.Report report)
        {
            if(string.IsNullOrEmpty(hash) || AcousticTake.HashFile(path)!=hash) throw new InvalidDataException("WAV hash mismatch: "+path);
            reader=new BinaryReader(File.OpenRead(path));
            try
            {
                if(new string(reader.ReadChars(4))!="RIFF" || reader.ReadUInt32()!=reader.BaseStream.Length-8 ||
                   new string(reader.ReadChars(8))!="WAVEfmt " || reader.ReadInt32()!=16 || reader.ReadUInt16()!=3)
                    throw new InvalidDataException("Expected the renderer's float32 WAV layout.");
                Channels=reader.ReadUInt16(); Rate=reader.ReadInt32(); int bytesPerSecond=reader.ReadInt32();
                int align=reader.ReadUInt16(),bits=reader.ReadUInt16(); string chunk=new string(reader.ReadChars(4)); long bytes=reader.ReadUInt32();
                if((Channels!=1 && Channels!=2) || Rate<=0 || align!=Channels*4 || bytesPerSecond!=(long)Rate*align || bits!=32 ||
                   chunk!="data" || bytes!=reader.BaseStream.Length-44 || bytes%align!=0)
                    throw new InvalidDataException("Invalid rendered WAV header or length.");
                Frames=bytes/align;
                int expectedChannels=report.formatVersion==1 ? 1:report.channelsPerListener;
                if(Rate!=report.sampleRate || Channels!=expectedChannels || Frames!=report.outputSamples) throw new InvalidDataException("WAV does not match its run report.");
            }
            catch {reader.Dispose(); throw;}
        }
        public float[] Read(long first,int count)
        {
            if(first<0 || count<1 || count>MaximumPreviewFrames || first>Frames-count) throw new ArgumentOutOfRangeException(nameof(count),"Choose a preview within the recording and at most 2,000,000 sample frames.");
            reader.BaseStream.Position=checked(44+first*Channels*4);
            var output=new float[checked(count*Channels)];
            for(int i=0;i<output.Length;i++) {output[i]=reader.ReadSingle(); if(!float.IsFinite(output[i])) throw new InvalidDataException("Non-finite audio sample.");}
            return output;
        }
        public void Dispose() => reader.Dispose();
    }

    public static TakeComparison.Report OpenReport(string path)
    {
        var r=JsonUtility.FromJson<TakeComparison.Report>(File.ReadAllText(path));
        if(r==null || !r.complete || (r.experiment!="direct-temporal-resolution-v1" && r.experiment!="spectral-temporal-resolution-v1")) throw new InvalidDataException("Select a completed temporal comparison.json.");
        foreach(var run in new[]{r.low,r.high})
        {
            if(run==null || !run.complete || (run.formatVersion<1 || run.formatVersion>3) || run.sampleRate<=0 || run.outputSamples<=0 || run.listenerCount<=0 ||
               run.audioHashes==null || run.audioHashes.Length!=run.listenerCount || string.IsNullOrEmpty(run.sceneHash) ||
               run.sceneHash!=r.sceneHash || run.takeStateHash!=r.stateHash || run.takeEventHash!=r.eventHash)
                throw new InvalidDataException("Comparison contains incomplete, unsupported or mismatched runs.");
            TakeRender.Validate(run.settings);
            if(run.formatVersion>=2 && (run.channelsPerListener!=run.settings.ChannelsPerListener || (run.channelsPerListener==2 && (string.IsNullOrEmpty(run.hrtfHash) || run.hrtfTaps<=0))))
                throw new InvalidDataException("Run channel/HRTF metadata does not match its renderer settings.");
        }
        var a=r.low; var b=r.high;
        var shape=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(a.settings)); shape.responseStepSamples=b.settings.responseStepSamples;
        if(a.sampleRate!=b.sampleRate || a.outputSamples!=b.outputSamples || a.listenerCount!=b.listenerCount || a.channelsPerListener!=b.channelsPerListener ||
           a.hrtfHash!=b.hrtfHash || a.hrtfTaps!=b.hrtfTaps || a.formatVersion!=b.formatVersion ||
           a.settings.responseStepSamples<=b.settings.responseStepSamples || JsonUtility.ToJson(shape)!=JsonUtility.ToJson(b.settings) ||
           r.listeners==null || r.listeners.Length!=a.listenerCount)
            throw new InvalidDataException("Low/high timelines, output formats or settings do not align.");
        foreach(var m in r.listeners) if(m==null || m.samples!=a.outputSamples || m.channels!=(a.formatVersion==1 ? 1:a.channelsPerListener) || !double.IsFinite(m.rmsError) || !double.IsFinite(m.maximumAbsoluteError) || !double.IsFinite(m.referenceRms) ||
            m.rmsError<0 || m.maximumAbsoluteError<0 || m.referenceRms<0 || (m.snrDefined && !double.IsFinite(m.snrDb))) throw new InvalidDataException("Invalid comparison metrics.");
        return r;
    }

    public static Vector2[] Envelope(float[] pcm,int channels,int ear,int columns)
    {
        int frames=pcm.Length/channels; columns=Math.Min(columns,frames); var values=new Vector2[columns];
        for(int x=0;x<columns;x++)
        {
            int first=(int)((long)x*frames/columns),end=(int)((long)(x+1)*frames/columns);
            float low=float.PositiveInfinity,high=float.NegativeInfinity;
            for(int n=first;n<end;n++) {float v=pcm[n*channels+ear]; low=Mathf.Min(low,v); high=Mathf.Max(high,v);}
            values[x]=new Vector2(low,high);
        }
        return values;
    }

    public sealed class Spectrum
    {
        public int Width,Height,Hop,Fft,Frames;
        public float[] Db;
    }
    // Display analysis only: Hann, one-sided peak-amplitude dBFS, linear frequency.
    // Temporal columns are decimated when necessary; frequency rows retain the largest bin.
    public static Spectrum Spectrogram(float[] pcm,int channels,int ear,int fft,int requestedHop)
    {
        if(fft<64 || fft>16384 || (fft&(fft-1))!=0 || requestedHop<1 || requestedHop>fft)
            throw new ArgumentException("Display FFT must be a power of two from 64 to 16384; hop must be 1..FFT.");
        int frames=pcm.Length/channels,hop=Math.Max(requestedHop,(int)Math.Ceiling((double)frames/512/requestedHop)*requestedHop);
        int width=(frames+hop-1)/hop,height=Math.Min(256,fft/2+1),bins=fft/2+1;
        var result=new Spectrum {Width=width,Height=height,Hop=hop,Fft=fft,Frames=frames,Db=new float[width*height]};
        var real=new double[fft]; var imag=new double[fft]; var window=new double[fft]; double scale=0;
        for(int n=0;n<fft;n++) {window[n]=.5-.5*Math.Cos(2*Math.PI*n/fft); scale+=window[n];}
        for(int x=0;x<width;x++)
        {
            Array.Clear(imag,0,fft); int first=x*hop;
            for(int n=0;n<fft;n++) real[n]=first+n<frames ? pcm[(first+n)*channels+ear]*window[n]:0;
            Fft(real,imag);
            for(int y=0;y<height;y++)
            {
                double maximum=0;
                for(int k=y*bins/height;k<(y+1)*bins/height;k++)
                {double amplitude=Math.Sqrt(real[k]*real[k]+imag[k]*imag[k])/scale*(k==0 || k==fft/2 ? 1:2); maximum=Math.Max(maximum,amplitude);}
                result.Db[y*width+x]=(float)(20*Math.Log10(Math.Max(1e-12,maximum)));
            }
        }
        return result;
    }
    internal static void Fft(double[] real,double[] imag)
    {
        int n=real.Length;
        for(int i=1,j=0;i<n;i++)
        {
            int bit=n>>1; for(; (j&bit)!=0;bit>>=1) j^=bit; j^=bit;
            if(i<j) {(real[i],real[j])=(real[j],real[i]); (imag[i],imag[j])=(imag[j],imag[i]);}
        }
        for(int length=2;length<=n;length<<=1)
        {
            double angle=-2*Math.PI/length,stepR=Math.Cos(angle),stepI=Math.Sin(angle);
            for(int first=0;first<n;first+=length)
            {
                double wr=1,wi=0;
                for(int k=0;k<length/2;k++)
                {
                    int a=first+k,b=a+length/2; double r=real[b]*wr-imag[b]*wi,i=real[b]*wi+imag[b]*wr;
                    real[b]=real[a]-r; imag[b]=imag[a]-i; real[a]+=r; imag[a]+=i;
                    double next=wr*stepR-wi*stepI; wi=wr*stepI+wi*stepR; wr=next;
                }
            }
        }
    }
    public static Texture2D Texture(Spectrum spectrum,float floorDb)
    {
        var texture=new Texture2D(spectrum.Width,spectrum.Height,TextureFormat.RGBA32,false) {hideFlags=HideFlags.HideAndDontSave,filterMode=FilterMode.Point};
        var pixels=new Color32[spectrum.Db.Length];
        for(int i=0;i<pixels.Length;i++)
        {
            float v=Mathf.Clamp01((spectrum.Db[i]-floorDb)/-floorDb);
            pixels[i]=v<.5f ? Color.Lerp(new Color(.035f,.045f,.075f),new Color(.12f,.65f,.7f),v*2):Color.Lerp(new Color(.12f,.65f,.7f),new Color(1,.88f,.35f),(v-.5f)*2);
        }
        texture.SetPixels32(pixels); texture.Apply(false,true); return texture;
    }
    public static float[] Audition(float[] low,float[] high,int channels,int first,int mode,float gain,int rate)
    {
        int frames=low.Length/channels;
        if(high.Length!=low.Length || first<0 || first>=frames || mode<0 || mode>2 || !float.IsFinite(gain) || gain<0) throw new ArgumentException("Invalid audition range.");
        var output=new float[(frames-first)*channels]; int fade=Math.Min(Math.Max(1,rate/200),(frames-first)/2);
        for(int n=first;n<frames;n++)
        {
            float weight=fade==0 ? 1:Mathf.Min(1,Mathf.Min((float)(n-first)/fade,(float)(frames-1-n)/fade));
            for(int c=0;c<channels;c++) {int at=n*channels+c; output[(n-first)*channels+c]=(mode==0 ? low[at]:mode==1 ? high[at]:low[at]-high[at])*gain*weight;}
        }
        return output;
    }

    // Unity's preview API is internal. Resolve exact signatures once; failure only disables audition.
    public sealed class Preview : IDisposable
    {
        readonly MethodInfo play,stop,position,isPlaying;
        AudioClip clip;
        int origin;
        public bool Supported => play!=null && stop!=null && position!=null && isPlaying!=null;
        public bool Active {get;private set;}
        public int Cursor {get;private set;}
        public Preview()
        {
            var type=typeof(Editor).Assembly.GetType("UnityEditor.AudioUtil"); const BindingFlags flags=BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Static;
            play=type?.GetMethod("PlayPreviewClip",flags,null,new[]{typeof(AudioClip),typeof(int),typeof(bool)},null);
            stop=type?.GetMethod("StopAllPreviewClips",flags,null,Type.EmptyTypes,null);
            position=type?.GetMethod("GetPreviewClipSamplePosition",flags,null,Type.EmptyTypes,null);
            isPlaying=type?.GetMethod("IsPreviewClipPlaying",flags,null,Type.EmptyTypes,null);
        }
        public int Poll()
        {
            if(Active)
            {
                if((bool)isPlaying.Invoke(null,null)) Cursor=origin+Math.Max(0,(int)position.Invoke(null,null));
                else {Cursor=origin+clip.samples; Active=false;}
            }
            return Cursor;
        }
        public void Play(float[] low,float[] high,int channels,int first,int mode,float gain,int rate)
        {
            if(!Supported) throw new NotSupportedException("Audio preview is unavailable in this Unity version. WAV inspection/export remains available.");
            Dispose(); origin=Cursor=first;
            var pcm=Audition(low,high,channels,first,mode,gain,rate);
            clip=AudioClip.Create("Acoustic comparison",pcm.Length/channels,channels,rate,false);
            clip.hideFlags=HideFlags.HideAndDontSave;
            try {if(!clip.SetData(pcm,0)) throw new InvalidOperationException("Cannot prepare preview clip."); play.Invoke(null,new object[]{clip,0,false}); Active=true;}
            catch {Dispose(); throw;}
        }
        public void Pause() {Poll(); if(Active) stop.Invoke(null,null); Active=false;}
        public void Dispose()
        {
            if(Active) stop.Invoke(null,null); Active=false;
            if(clip!=null) {UnityEngine.Object.DestroyImmediate(clip); clip=null;}
        }
    }
}
