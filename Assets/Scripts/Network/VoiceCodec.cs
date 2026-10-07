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
        readonly Action<int,byte[],int> send;
        int fill;
        public int Frames {get;private set;}
        public Encoder(Action<int,byte[],int> send) {this.send=send;codec.Bitrate=32000;codec.Complexity=5;codec.UseDTX=false;}
        public void Append(float[] input,int count)
        {
            for(int offset=0;offset<count;)
            {
                int n=Math.Min(Frame-fill,count-offset);Array.Copy(input,offset,frame,fill,n);offset+=n;fill+=n;
                if(fill==Frame)Emit();
            }
        }
        public void Finish() {if(fill>0) {Array.Clear(frame,fill,Frame-fill);Emit();}}
        void Emit() {int bytes=codec.Encode(frame.AsSpan(),Frame,packet.AsSpan(),packet.Length);send(Frames++,packet,bytes);fill=0;}
    }
    internal sealed class Receiver : IDisposable
    {
        readonly OpusDecoder codec=new OpusDecoder(Rate,1);
        readonly Dictionary<int,byte[]> pending=new Dictionary<int,byte[]>();
        readonly float[] decoded=new float[Frame];
        readonly PcmRateConverter converter;
        readonly Action<float[],int,SourceStream.Status> output;
        readonly int maximumFuture;
        SourceStream.Status status;
        public int Next {get;private set;}
        public int Late {get;private set;}
        public int Concealed {get;private set;}
        public int Final {get;private set;}=int.MaxValue;
        public Receiver(int outputRate,Action<float[],int,SourceStream.Status> output,int firstFrame=0,int maximumFuture=100)
        {if(firstFrame<0 || maximumFuture<1)throw new ArgumentOutOfRangeException();Next=firstFrame;this.maximumFuture=maximumFuture;this.output=output;converter=new PcmRateConverter(Rate,outputRate,(data,count)=>this.output(data,count,status));}
        public bool Add(int sequence,byte[] packet)
        {
            if(sequence<Next) {Late++;return false;}
            // Bound queued future packets; this does not cap recording length.
            if(sequence>=Final || sequence-Next>=maximumFuture || packet==null || packet.Length==0 || packet.Length>MaximumPacket || pending.ContainsKey(sequence))return false;
            if(OpusPacketInfo.GetNumSamples(packet.AsSpan(),Rate)!=Frame)return false;
            pending.Add(sequence,packet);return true;
        }
        public void Stop(int frames) {if(frames<0)throw new ArgumentOutOfRangeException(nameof(frames));Final=Math.Min(Final,frames);}
        public bool DecodeNext()
        {
            if(Next>=Final) {converter.Finish();return false;}
            bool received=pending.TryGetValue(Next,out var packet);pending.Remove(Next++);
            status=received?SourceStream.Status.Data:SourceStream.Status.Concealed;
            int count;
            try {count=codec.Decode(received?packet.AsSpan():ReadOnlySpan<byte>.Empty,decoded.AsSpan(),Frame,false);}
            catch(Concentus.OpusException) {status=SourceStream.Status.Concealed;count=codec.Decode(ReadOnlySpan<byte>.Empty,decoded.AsSpan(),Frame,false);}
            if(status==SourceStream.Status.Concealed)Concealed++;
            if(count!=Frame)throw new InvalidOperationException("Voice packets must contain exactly 20 ms.");
            converter.Process(decoded,count);return true;
        }
        public void Dispose() {converter.Dispose();pending.Clear();}
    }
}
