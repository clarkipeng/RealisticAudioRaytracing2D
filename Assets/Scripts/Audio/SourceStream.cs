using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;

// Absolute source samples. Only the producer thread reads/writes this bounded history.
public sealed class SourceStream : IDisposable
{
    public enum Status : uint { Data, Silence, Missing, Concealed, Discontinuity=4 }
    public long First {get;private set;}
    public long End {get;private set;}
    public bool Finished {get;private set;}
    public int Capacity=>ring.Length;
    readonly float[] ring;
    readonly BinaryReader file;
    long page=-1;
    int pageCount;
    public Action<long,float[],int,int,long,Status> Recorded;
    public SourceStream(int capacity) {if(capacity<1)throw new ArgumentOutOfRangeException(nameof(capacity));ring=new float[capacity];}
    public SourceStream(string path,long count)
    {
        if(new FileInfo(path).Length!=checked(count*4))throw new InvalidDataException("Stream PCM length mismatch.");
        file=new BinaryReader(File.OpenRead(path));ring=new float[16384];End=count;Finished=true;
    }
    public void Append(float[] data,int offset,int count,long availableAt,Status status=Status.Data)
    {
        if(Finished || count<0 || availableAt<0 || (uint)status>7 || (data!=null && (offset<0 || count>data.Length-offset)))
            throw new ArgumentException("Invalid source block.");
        if(count>ring.Length-(End-First))throw new InvalidOperationException("Source history overrun; the renderer is not consuming its inputs.");
        for(int i=0;i<count;i++)
        {
            float value=data==null?0:data[offset+i];
            if(!float.IsFinite(value))throw new InvalidDataException("Non-finite source PCM.");
        }
        Recorded?.Invoke(End,data,offset,count,availableAt,status);
        for(int i=0;i<count;i++)ring[(int)((End+i)%ring.Length)]=data==null?0:data[offset+i];
        End+=count;
    }
    public void SilenceUntil(long end,long availableAt,Status status=Status.Silence)
    {
        while(End<end)Append(null,0,(int)Math.Min(4096,end-End),availableAt,status);
    }
    public void Finish()=>Finished=true;
    public void Prune(long before) {if(file==null)First=Math.Max(First,Math.Min(End,Math.Max(0,before)));}
    public float Read(long sample)
    {
        if(sample<0 || (Finished && sample>=End))return 0;
        if(sample<First || sample>=End)throw new InvalidOperationException("Unfinalized or expired source sample: "+sample);
        if(file==null)return ring[(int)(sample%ring.Length)];
        long start=sample/ring.Length*ring.Length;
        if(page!=start)
        {
            page=start;pageCount=(int)Math.Min(ring.Length,End-start);file.BaseStream.Position=start*4;
            for(int i=0;i<pageCount;i++) {ring[i]=file.ReadSingle();if(!float.IsFinite(ring[i]))throw new InvalidDataException("Non-finite recorded source PCM.");}
        }
        return ring[(int)(sample-page)];
    }
    public void Dispose()=>file?.Dispose();

    // Bounded disk queue. No file IO or waiting in the audio callback; failures stay incomplete.
    internal sealed class Capture : IDisposable
    {
        struct Block {public int source,count;public long sample,available;public Status status;public byte[] data;public TaskCompletionSource<bool> flushed;}
        readonly BlockingCollection<Block> queue=new BlockingCollection<Block>(128);
        readonly Task writer;
        readonly BinaryWriter[] pcm;
        readonly BinaryWriter blocks;
        readonly string directory;
        readonly AcousticTake.StreamDefinition[] descriptions;
        readonly SourceStream[] streams;
        Task cleanup;
        volatile Exception failure;
        public Exception Failure=>failure??writer.Exception?.GetBaseException();
        public Capture(string directory,SourceStream[] streams,AcousticTake.Manifest manifest)
        {
            this.directory=directory;this.streams=streams;pcm=new BinaryWriter[streams.Length];descriptions=new AcousticTake.StreamDefinition[streams.Length];
            try
            {
                blocks=new BinaryWriter(File.Open(Path.Combine(directory,"source-blocks.bin"),FileMode.CreateNew));
                for(int i=0;i<streams.Length;i++)if(streams[i]!=null)
                {
                    int source=i;
                    descriptions[i]=new AcousticTake.StreamDefinition {sourceIndex=i,file="source-"+i+".f32"};
                    pcm[i]=new BinaryWriter(File.Open(Path.Combine(directory,descriptions[i].file),FileMode.CreateNew));
                    streams[i].Recorded=(sample,data,offset,count,available,status)=>Enqueue(source,sample,data,offset,count,available,status);
                }
                manifest.streams=descriptions;
                writer=Task.Run(Write);
            }
            catch {foreach(var stream in streams)if(stream!=null)stream.Recorded=null;foreach(var p in pcm)p?.Dispose();blocks?.Dispose();queue.Dispose();throw;}
        }
        void Enqueue(int source,long sample,float[] data,int offset,int count,long available,Status status)
        {
            if(Failure!=null)return;
            var copy=data==null?null:ArrayPool<byte>.Shared.Rent(checked(count*4));
            if(copy!=null)
            {
                Buffer.BlockCopy(data,offset*4,copy,0,count*4);
                if(!BitConverter.IsLittleEndian)for(int i=0;i<count;i++)Array.Reverse(copy,i*4,4);
            }
            if(!queue.TryAdd(new Block {source=source,sample=sample,count=count,data=copy,available=available,status=status}))
            {Return(copy);failure=new IOException("Source recording queue overflow; this take is incomplete.");}
        }
        static void Return(byte[] data) {if(data!=null)ArrayPool<byte>.Shared.Return(data);}
        void Write()
        {
            try
            {
                foreach(var b in queue.GetConsumingEnumerable())
                {
                    if(b.flushed!=null){b.flushed.SetResult(true);continue;}
                    try {
                    if(b.sample!=descriptions[b.source].samples)throw new InvalidDataException("Non-contiguous recorded source.");
                    if(b.data==null) {var f=pcm[b.source].BaseStream;f.Position+=b.count*4L;f.SetLength(f.Position);}
                    else pcm[b.source].Write(b.data,0,b.count*4);
                    blocks.Write(b.source);blocks.Write(b.sample);blocks.Write(b.count);blocks.Write(b.available);blocks.Write((uint)b.status);
                    descriptions[b.source].samples+=b.count;
                    } finally {Return(b.data);}
                }
            }
            finally {foreach(var p in pcm)p?.Dispose();blocks.Dispose();}
        }
        public void Complete(AcousticTake.Manifest manifest)
        {
            Dispose();if(Failure!=null)throw new IOException("Source recording failed.",Failure);
            foreach(var d in descriptions)if(d!=null)d.sha256=AcousticTake.HashFile(Path.Combine(directory,d.file));
            manifest.sourceBlocksHash=AcousticTake.HashFile(Path.Combine(directory,"source-blocks.bin"));
        }
        // Offline callers may wait for disk; Recorded/Enqueue never waits on an audio producer.
        public void Flush()
        {
            if(Failure!=null)throw new IOException("Source recording failed.",Failure);
            var flushed=new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            if(!queue.TryAdd(new Block{flushed=flushed}))throw new IOException("Source recording queue is full.");
            Task.WhenAny(flushed.Task,writer).GetAwaiter().GetResult();
            if(Failure!=null||!flushed.Task.IsCompleted)throw new IOException("Source recording stopped before its flush.",Failure);
        }
        // Stop accepting blocks immediately; failed recording must not stall live audio.
        public Task Abort()
        {
            if(cleanup!=null)return cleanup;
            foreach(var stream in streams)if(stream!=null)stream.Recorded=null;
            queue.CompleteAdding();return cleanup=Finish();
        }
        async Task Finish()
        {
            try {await writer.ConfigureAwait(false);}catch(Exception e) {failure=failure??e;}
            finally {while(queue.TryTake(out var b))Return(b.data);queue.Dispose();}
        }
        public void Dispose()=>Abort().GetAwaiter().GetResult();
    }
}
