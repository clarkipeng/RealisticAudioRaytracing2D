using System;
using System.Collections.Generic;
using Concentus.Enums;
using Concentus.Structs;

// One mono 48 kHz, 20 ms Opus stream per speaking epoch. Packet order is independent of transport order.
internal static class VoiceCodec
{
    public const int Rate=48000,Frame=960,MaximumPacket=1200;
    internal sealed class Encoder
    {
        readonly OpusEncoder codec=new OpusEncoder(Rate,1,OpusApplication.OPUS_APPLICATION_VOIP);
        readonly float[] frame=new float[Frame];
        readonly byte[] packet=new byte[MaximumPacket];
        readonly Action<long,byte[],int> send;
        int fill;
        bool finished;
        public long NextSample {get;private set;}
        public long InputSamples {get;private set;}
        public int DelaySamples=>codec.Lookahead;
        public Encoder(Action<long,byte[],int> send) {this.send=send;codec.Bitrate=32000;codec.Complexity=5;codec.UseDTX=false;}
        public void Append(float[] input,int count)
        {
            if(finished || count<0 || count>input.Length)throw new ArgumentException("Invalid encoder input.");
            InputSamples+=count;Encode(input,count);
        }
        void Encode(float[] input,int count)
        {
            for(int offset=0;offset<count;)
            {
                int n=Math.Min(Frame-fill,count-offset);Array.Copy(input,offset,frame,fill,n);offset+=n;fill+=n;
                if(fill==Frame)Emit();
            }
        }
        public void Finish()
        {
            if(finished)return;finished=true;
            if(InputSamples==0)return;
            // Flush the codec's delayed final input, then pad only the last encoded frame.
            Encode(new float[DelaySamples],DelaySamples);
            if(fill>0) {Array.Clear(frame,fill,Frame-fill);Emit();}
        }
        void Emit() {int bytes=codec.Encode(frame.AsSpan(),Frame,packet.AsSpan(),packet.Length);send(NextSample,packet,bytes);NextSample+=Frame;fill=0;}
    }
    internal sealed class Receiver : IDisposable
    {
        readonly OpusDecoder codec=new OpusDecoder(Rate,1);
        readonly Dictionary<long,byte[]> pending=new Dictionary<long,byte[]>();
        readonly float[] decoded=new float[Frame];
        readonly PcmRateConverter converter;
        readonly Action<float[],int,SourceStream.Status> output;
        readonly int maximumFuture,delay;
        SourceStream.Status status;
        long lastCorrection=-Rate;
        public long Next {get;private set;}
        public long Sample=>Math.Max(0,Next-delay);
        public double PendingSamples=>converter.PendingSamples;
        public long Late {get;private set;}
        public long Concealed {get;private set;}
        public long Final {get;private set;}=long.MaxValue;
        public Receiver(int outputRate,Action<float[],int,SourceStream.Status> output,int delaySamples=0,long firstSample=0,int maximumFuture=100)
        {
            if(firstSample<0 || firstSample%Frame!=0 || maximumFuture<1 || delaySamples<0 || delaySamples>=Rate)throw new ArgumentOutOfRangeException();
            Next=firstSample;delay=delaySamples;this.maximumFuture=maximumFuture;this.output=output;
            converter=new PcmRateConverter(Rate,outputRate,(data,count)=>this.output(data,count,status),variableRate:true);
        }
        public bool Add(long sample,byte[] packet)
        {
            if(sample<Next) {Late++;return false;}
            // Bound queued future packets; this does not cap recording length.
            if(sample%Frame!=0 || (Final!=long.MaxValue && sample>=Final+delay) || sample-Next>=(long)maximumFuture*Frame || packet==null || packet.Length==0 || packet.Length>MaximumPacket || pending.ContainsKey(sample))return false;
            if(OpusPacketInfo.GetNumSamples(packet.AsSpan(),Rate)!=Frame)return false;
            pending.Add(sample,packet);return true;
        }
        public void Stop(long validSamples) {if(validSamples<0 || validSamples>long.MaxValue-Rate)throw new ArgumentOutOfRangeException(nameof(validSamples));Final=Math.Min(Final,validSamples);}
        public void CorrectClock(double errorSeconds)
        {
            if(Next-lastCorrection<Rate)return;lastCorrection=Next;
            // Spread phase error over two seconds; at most 0.5% rate correction, never insert/drop a block.
            converter.SetInputRate(Rate/(1+Math.Clamp(errorSeconds/2,-.005,.005)));
        }
        public bool DecodeNext()
        {
            if(Sample>=Final) {converter.Finish();return false;}
            bool received=pending.TryGetValue(Next,out var packet);pending.Remove(Next);
            status=received?SourceStream.Status.Data:SourceStream.Status.Concealed;
            int count;
            try {count=codec.Decode(received?packet.AsSpan():ReadOnlySpan<byte>.Empty,decoded.AsSpan(),Frame,false);}
            catch(Concentus.OpusException) {status=SourceStream.Status.Concealed;count=codec.Decode(ReadOnlySpan<byte>.Empty,decoded.AsSpan(),Frame,false);}
            if(status==SourceStream.Status.Concealed)Concealed++;
            if(count!=Frame)throw new InvalidOperationException("Voice packets must contain exactly 20 ms.");
            int skip=(int)Math.Min(Frame,Math.Max(0,delay-Next));
            count=(int)Math.Min(Frame-skip,Final-Sample);Next+=Frame;
            if(count>0) {Array.Copy(decoded,skip,decoded,0,count);converter.Process(decoded,count);}
            return true;
        }
        public void Dispose() {converter.Dispose();pending.Clear();}
    }
}
