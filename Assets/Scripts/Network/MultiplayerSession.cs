using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Services.Core;
using Unity.Services.Authentication;
using Unity.Services.Relay;
using Unity.Services.Relay.Models;
using Unity.Collections;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using UnityEngine;

// Shared-world and dry-voice transport adapter. Acoustic capture stays local to each client, after presentation/voice decoding.
[DefaultExecutionOrder(-10)]
public sealed class MultiplayerSession : MonoBehaviour
{
    public SceneSession session;
    public MicrophoneInput microphone;
    public RaytracedAudioSource[] remoteVoices=Array.Empty<RaytracedAudioSource>();
    public Transform[] spawnPoints=Array.Empty<Transform>();
    public Transform[] sharedObjects=Array.Empty<Transform>();
    public string address="127.0.0.1";
    public ushort port=7777;
    [Min(.02f)] public float presentationSeconds=.1f;
    public bool Connected => net!=null && net.IsConnectedClient;
    public bool IsHost => net!=null && net.IsServer;
    public bool ControlsWorld => net==null || IsHost;
    public int LocalSlot {get;private set;}
    public string Status {get;private set;}="Solo";
    public bool Ready {get;private set;}
    public bool Connecting => !Ready && !closing;
    public int ReceivedPackets {get;private set;}
    public long ConcealedFrames {get;private set;}
    public long LatePackets {get;private set;}
    public int ConnectedPlayers=>IsHost?clients.Count:shown==null?0:shown.Count(value=>value);
    public int Capacity=>remoteVoices.Length+1;
    public static string PendingMode,PendingAddress;
    public string JoinCode {get;private set;}
    public bool UsesRelay {get;private set;}=true;
    public double SessionTime=>Connected?(IsHost?net.NetworkTimeSystem.LocalTime:net.NetworkTimeSystem.LocalTime-net.NetworkTimeSystem.LocalBufferSec):Time.realtimeSinceStartupAsDouble;
    NetworkManager net;
    readonly Dictionary<ulong,int> clients=new Dictionary<ulong,int>();
    ulong[] occupants;
    bool[] present;
    Pose[] poses;
    double[] poseTimes;
    int localEpoch;
    int[] poseEpochs,shownEpochs;
    readonly List<PoseFrame> frames=new List<PoseFrame>();
    Pose[] presented;
    bool[] shown;
    VoicePeer[] voices;
    VoiceCodec.Encoder encoder;
    int voiceEpoch;
    double voiceStart,hostOrigin,lastSend;
    string sessionId,fingerprint;
    bool replay,closing;
    const string Message="acoustics-v2";
    enum Kind : byte {Welcome,Pose,World,Interact,VoiceStart,VoicePacket,VoiceStop}
    sealed class PoseFrame {public double time;public Pose[] poses;public double[] times;public bool[] present;public int[] epochs;}
    sealed class VoicePeer
    {
        public int epoch;
        public double time,extraDelay;
        public int delay;
        public long firstSample,lastPacket=-1,final=long.MaxValue;
        public SourceStream bound;
        public SampleClock clock=new SampleClock(VoiceCodec.Rate);
        public VoiceCodec.Receiver decoder;
        public VoicePeer next;
        public readonly List<(long sample,double time,byte[] data)> pending=new List<(long,double,byte[])>();
    }
    public void Prepare(AcousticTake.SessionDefinition saved=null)
    {
        replay=saved!=null;LocalSlot=saved?.localSlot??0;
        if(saved!=null && saved.slots!=Capacity)throw new InvalidDataException("Recorded roster does not match this scene.");
        poseTimes=new double[Capacity];occupants=new ulong[Capacity];present=new bool[Capacity];poses=new Pose[Capacity];voices=new VoicePeer[Capacity];presented=new Pose[Capacity];shown=new bool[Capacity];poseEpochs=new int[Capacity];shownEpochs=new int[Capacity];
        for(int i=0;i<Capacity;i++) {occupants[i]=ulong.MaxValue;poses[i]=new Pose(session.player.position,session.manager.takeListeners[session.manager.monitorListener].rotation);voices[i]=new VoicePeer();}
        foreach(var voice in remoteVoices)voice.transform.parent.gameObject.SetActive(false);
        microphone.source.mutedListenerPlusOne=microphone.monitorSelf?0:1;
        microphone.Clock=()=>SessionTime;
        microphone.VoiceState+=LocalVoiceState;microphone.VoiceSamples+=LocalVoiceSamples;
        Ready=replay;Status=replay?"Replay (offline)":"Solo";
    }
    public IEnumerator Connect()
    {
        string mode=PendingMode;PendingMode=null;
        UsesRelay=string.IsNullOrEmpty(mode) || mode.StartsWith("relay-",StringComparison.Ordinal);
        if(string.IsNullOrEmpty(mode)) {Ready=true;yield break;}
        if(PendingAddress!=null)address=PendingAddress;PendingAddress=null;
        Status="Connecting";
        var connection=StartConnection(mode);
        while(!connection.IsCompleted && !closing)yield return null;
        if(closing)yield break;
        if(connection.IsFaulted || connection.IsCanceled)
        {Status="Connection failed: "+connection.Exception?.GetBaseException().Message;Shutdown();yield break;}
        double deadline=Time.realtimeSinceStartupAsDouble+15;
        while(!Ready && net!=null && Time.realtimeSinceStartupAsDouble<deadline)yield return null;
        if(!Ready) {Status="Connection failed: "+(net?.DisconnectReason??"timeout");Shutdown();yield break;}
        session.player.SetPositionAndRotation(spawnPoints[LocalSlot].position,spawnPoints[LocalSlot].rotation);Physics.SyncTransforms();
        Status=JoinCode!=null?(IsHost?"Hosting — code ":"Joined — code ")+JoinCode:IsHost?"Hosting LAN :"+port:"Connected to "+address;
    }
    async Task StartConnection(string mode)
    {
        if(spawnPoints.Length!=Capacity || Array.Exists(spawnPoints,p=>p==null))throw new InvalidOperationException("Assign one spawn point per player slot.");
        bool host=mode=="host" || mode=="relay-host",relay=mode.StartsWith("relay-",StringComparison.Ordinal);
        var go=new GameObject("Acoustic transport");net=go.AddComponent<NetworkManager>();
        var transport=go.AddComponent<UnityTransport>();
        net.NetworkConfig=new NetworkConfig {NetworkTransport=transport,EnableSceneManagement=false,ConnectionApproval=true,
            ProtocolVersion=4,TickRate=30,ForceSamePrefabs=false};
        fingerprint=BuildFingerprint()+"|"+gameObject.scene.path+"|"+Capacity;
        net.NetworkConfig.ConnectionData=Encoding.UTF8.GetBytes(fingerprint);
        net.ConnectionApprovalCallback=Approve;
        net.OnClientConnectedCallback+=Joined;net.OnClientDisconnectCallback+=Left;
        if(relay)
        {
            if(string.IsNullOrEmpty(Application.cloudProjectId))throw new InvalidOperationException("Link this project in Project Settings > Services to use internet hosting.");
            if(!host && string.IsNullOrWhiteSpace(address))throw new InvalidOperationException("Enter the host's join code.");
            // Independent local players need independent anonymous identities.
            await UnityServices.InitializeAsync(new InitializationOptions().SetProfile("acoustics-"+System.Diagnostics.Process.GetCurrentProcess().Id));
            if(!AuthenticationService.Instance.IsSignedIn)await AuthenticationService.Instance.SignInAnonymouslyAsync();
            if(closing)return;
            if(host)
            {
                var allocation=await RelayService.Instance.CreateAllocationAsync(Capacity-1);
                JoinCode=await RelayService.Instance.GetJoinCodeAsync(allocation.AllocationId);
                if(closing)return;
                transport.SetRelayServerData(AllocationUtils.ToRelayServerData(allocation,"dtls"));
            }
            else
            {
                JoinCode=address.Trim().ToUpperInvariant();
                var allocation=await RelayService.Instance.JoinAllocationAsync(JoinCode);
                if(closing)return;
                transport.SetRelayServerData(AllocationUtils.ToRelayServerData(allocation,"dtls"));
            }
        }
        else transport.SetConnectionData(address,port,"0.0.0.0");
        if(!(host?net.StartHost():net.StartClient()))throw new InvalidOperationException("Could not start network transport.");
        net.CustomMessagingManager.RegisterNamedMessageHandler(Message,Receive);
    }
    void Approve(NetworkManager.ConnectionApprovalRequest request,NetworkManager.ConnectionApprovalResponse response)
    {
        int free=Array.FindIndex(occupants,x=>x==ulong.MaxValue);
        response.Approved=Encoding.UTF8.GetString(request.Payload)==fingerprint && free>=0;
        response.CreatePlayerObject=false;
        if(response.Approved) {clients.Add(request.ClientNetworkId,free);occupants[free]=request.ClientNetworkId;}
        else response.Reason="Use the same scene/build; all configured player slots may be occupied.";
    }
    void Joined(ulong id)
    {
        if(!IsHost)return;
        int slot=clients[id];present[slot]=true;poses[slot]=new Pose(spawnPoints[slot].position,session.manager.takeListeners[session.manager.monitorListener].rotation);
        if(id==net.LocalClientId) {sessionId=Guid.NewGuid().ToString("N");hostOrigin=SessionTime;LocalSlot=slot;Ready=true;}
        else
        {
            Send(id,Kind.Welcome,w=> {w.Write(slot);w.Write(sessionId);w.Write(hostOrigin);},true);
            for(int source=0;source<Capacity;source++)
            {
                int current=source;
                if(source==LocalSlot && encoder!=null)
                    Send(id,Kind.VoiceStart,w=> {w.Write(current);w.Write(voiceEpoch);w.Write(microphone.Active?microphone.CaptureTime(0):voiceStart);w.Write(encoder.NextSample);w.Write(encoder.DelaySamples);},true);
                else for(var voice=voices[source];voice!=null;voice=voice.next)
                {
                    if(voice.decoder==null || voice.decoder.Sample>=voice.decoder.Final)continue;
                    Send(id,Kind.VoiceStart,w=> {w.Write(current);w.Write(voice.epoch);w.Write(voice.clock.At(0));w.Write(voice.decoder.Next);w.Write(voice.delay);},true);
                    if(voice.final!=long.MaxValue)Send(id,Kind.VoiceStop,w=> {w.Write(current);w.Write(voice.epoch);w.Write(voice.final);},true);
                }
            }
        }
    }
    void Left(ulong id)
    {
        if(closing)return;
        if(IsHost && clients.TryGetValue(id,out int slot))
        {
            present[slot]=false;voices[slot].decoder?.Stop(voices[slot].decoder.Sample);clients.Remove(id);
            // Do not reuse a participant slot inside a recording: delayed packets/tails retain their identity.
        }
        if(net!=null && id==net.LocalClientId) {Status="Disconnected: "+net.DisconnectReason;session.SaveRecording();}
    }
    public AcousticTake.SessionDefinition Description()=>new AcousticTake.SessionDefinition {scene=gameObject.scene.path,
        sessionId=sessionId??Guid.NewGuid().ToString("N"),build=fingerprint,clientId=Connected?net.LocalClientId.ToString():"local",mode=Connected?"lan-client-presented-v1":"solo",
        slots=Capacity,localSlot=LocalSlot,serverTimeOrigin=Connected?SessionTime:0};
    public void CaptureStarted()
    {
        if(session.manager.SessionInfo!=null)session.manager.SessionInfo.serverTimeOrigin=
            SessionTime-(double)session.manager.LiveSample/session.manager.sampleRate;
    }
    public void TeleportPlayer(Vector3 position,Quaternion rotation)
    {
        localEpoch++;session.player.SetPositionAndRotation(position,rotation);session.Walker.Initialize();session.Look.Initialize();session.manager.MarkTakeDiscontinuity(session.player);Physics.SyncTransforms();
    }
    public void RequestInteraction() {if(Connected && !IsHost)Send(0,Kind.Interact,null,true);}
    public void Restart(string mode)
    {
        PendingMode=mode;PendingAddress=address;session.RestartScene();
    }
    RaytracedAudioSource Remote(int slot)=>remoteVoices[slot<LocalSlot?slot:slot-1];
    void Receive(ulong sender,FastBufferReader buffer)
    {
        if(replay || closing)return;
        try
        {
            int remaining=buffer.Length-buffer.Position;
            if(remaining>16384)throw new InvalidDataException("Oversized session message.");
            var bytes=new byte[remaining];buffer.ReadBytesSafe(ref bytes,bytes.Length);
            using var reader=new BinaryReader(new MemoryStream(bytes,false));
            var kind=(Kind)reader.ReadByte();
            if(kind==Kind.Welcome)
            {
                if(IsHost || sender!=0)return;
                LocalSlot=reader.ReadInt32();sessionId=reader.ReadString();hostOrigin=reader.ReadDouble();
                if(LocalSlot<0 || LocalSlot>=Capacity)throw new InvalidDataException("Invalid player slot.");Ready=true;
            }
            else if(kind==Kind.Pose)
            {
                if(!IsHost || !clients.TryGetValue(sender,out int slot))return;
                int epoch=reader.ReadInt32();double time=reader.ReadDouble();var pose=ReadPose(reader);
                if(!double.IsFinite(time))throw new InvalidDataException("Invalid pose time.");
                if(time>poseTimes[slot]) {poseEpochs[slot]=epoch;poseTimes[slot]=time;poses[slot]=pose;}
            }
            else if(kind==Kind.World)
            {
                if(IsHost || sender!=0)return;
                double time=reader.ReadDouble();if(!double.IsFinite(time))throw new InvalidDataException("Invalid network time.");
                var frame=new PoseFrame {time=time,poses=new Pose[Capacity+sharedObjects.Length],times=new double[Capacity],present=new bool[Capacity],epochs=new int[Capacity]};
                for(int i=0;i<Capacity;i++) {frame.present[i]=reader.ReadBoolean();frame.epochs[i]=reader.ReadInt32();frame.times[i]=reader.ReadDouble();if(!double.IsFinite(frame.times[i]))throw new InvalidDataException("Invalid pose time.");frame.poses[i]=ReadPose(reader);}
                for(int i=0;i<sharedObjects.Length;i++)frame.poses[Capacity+i]=ReadPose(reader);
                AddFrame(frame);
            }
            else if(kind==Kind.Interact)
            {
                if(IsHost && clients.TryGetValue(sender,out int slot))
                    foreach(var actor in session.actors)if(actor is SlidingDoor door)door.Interact(poses[slot].position);
            }
            else if(kind>=Kind.VoiceStart && kind<=Kind.VoiceStop)
            {
                int slot=reader.ReadInt32();int epoch=reader.ReadInt32();
                if(IsHost) {if(!clients.TryGetValue(sender,out int owned) || slot!=owned)return;}
                else if(sender!=0)return;
                if(slot<0 || slot>=Capacity || slot==LocalSlot)return;
                if(kind==Kind.VoiceStart)
                {
                    double time=reader.ReadDouble();if(!double.IsFinite(time))throw new InvalidDataException("Invalid voice time.");
                    StartVoice(slot,epoch,time,reader.ReadInt64(),reader.ReadInt32());
                }
                else if(kind==Kind.VoicePacket)
                {
                    long first=reader.ReadInt64();double time=reader.ReadDouble();int count=reader.ReadInt32();
                    if(first<0 || first>long.MaxValue-VoiceCodec.Rate || first%VoiceCodec.Frame!=0 || !double.IsFinite(time))throw new InvalidDataException("Invalid voice timestamp.");
                    if(count<1 || count>VoiceCodec.MaximumPacket || count!=reader.BaseStream.Length-reader.BaseStream.Position)throw new InvalidDataException("Invalid voice packet.");
                    var data=reader.ReadBytes(count);var voice=GetVoice(slot,epoch);
                    if(voice!=null)
                    {
                        if(voice.decoder!=null)AddVoicePacket(voice,first,time,data);
                        else if(voice.pending.Count<VoiceWindow)voice.pending.Add((first,time,data));
                    }
                }
                else
                {
                    long final=reader.ReadInt64();if(final<0 || final>long.MaxValue-VoiceCodec.Rate)throw new InvalidDataException("Invalid voice end.");
                    var voice=GetVoice(slot,epoch);if(voice!=null) {voice.final=Math.Min(voice.final,final);voice.decoder?.Stop(voice.final);}
                }
                if(IsHost)foreach(var client in clients.Keys)if(client!=net.LocalClientId && client!=sender)SendBytes(client,bytes,kind!=Kind.VoicePacket);
            }
            else throw new InvalidDataException("Unknown session message.");
        }
        catch(Exception e) when(e is IOException || e is InvalidDataException || e is ArgumentException || e is OverflowException)
        {Debug.LogWarning("Rejected session message: "+e.Message,this);}
    }
    static Pose ReadPose(BinaryReader r)
    {
        var position=new Vector3(r.ReadSingle(),r.ReadSingle(),r.ReadSingle());
        var q=new Quaternion(r.ReadSingle(),r.ReadSingle(),r.ReadSingle(),r.ReadSingle());
        float norm=Quaternion.Dot(q,q);
        if(!float.IsFinite(position.sqrMagnitude) || !float.IsFinite(norm) || norm<.9f || norm>1.1f)throw new InvalidDataException("Invalid network pose.");
        return new Pose(position,Quaternion.Normalize(q));
    }
    static void WritePose(BinaryWriter w,Pose p) {w.Write(p.position.x);w.Write(p.position.y);w.Write(p.position.z);w.Write(p.rotation.x);w.Write(p.rotation.y);w.Write(p.rotation.z);w.Write(p.rotation.w);}
    void Send(ulong client,Kind kind,Action<BinaryWriter> payload,bool reliable=false)
    {
        using var bytes=new MemoryStream();using var writer=new BinaryWriter(bytes);writer.Write((byte)kind);payload?.Invoke(writer);SendBytes(client,bytes.ToArray(),reliable);
    }
    void SendBytes(ulong client,byte[] bytes,bool reliable)
    {
        using var writer=new FastBufferWriter(bytes.Length,Allocator.Temp);writer.WriteBytesSafe(bytes);
        net.CustomMessagingManager.SendNamedMessage(Message,client,writer,reliable?NetworkDelivery.ReliableFragmentedSequenced:NetworkDelivery.Unreliable);
    }
    void Broadcast(Kind kind,Action<BinaryWriter> payload,bool reliable)
    {
        if(IsHost) {foreach(ulong client in clients.Keys)if(client!=net.LocalClientId)Send(client,kind,payload,reliable);}
        else if(Connected)Send(0,kind,payload,reliable);
    }
    void LocalVoiceState(bool active,double time)
    {
        if(!Connected)return;
        if(active)
        {
            voiceEpoch++;voiceStart=time;
            encoder=new VoiceCodec.Encoder((first,data,count)=>Broadcast(Kind.VoicePacket,w=>
            {
                w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(first);
                w.Write(microphone.Active?microphone.CaptureTime(first):voiceStart+(double)first/VoiceCodec.Rate);
                w.Write(count);w.Write(data,0,count);
            },false));
            Broadcast(Kind.VoiceStart,w=> {w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(time);w.Write(0L);w.Write(encoder.DelaySamples);},true);
        }
        else if(encoder!=null)
        {
            encoder.Finish();long final=encoder.InputSamples;encoder=null;
            Broadcast(Kind.VoiceStop,w=> {w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(final);},true);
        }
    }
    void LocalVoiceSamples(float[] data,int count)=>encoder?.Append(data,count);
    int VoiceWindow=>(int)Math.Ceiling((presentationSeconds+1)*50);
    VoicePeer GetVoice(int slot,int epoch)
    {
        var voice=voices[slot];if(epoch<1 || epoch<voice.epoch)return null;
        if(voice.epoch==0)return voices[slot]=new VoicePeer {epoch=epoch};
        // A restarted microphone must not discard the previous epoch's buffered final speech.
        for(int queued=0;queued<VoiceWindow;queued++)
        {
            if(epoch==voice.epoch)return voice;
            if(voice.next==null || voice.next.epoch>epoch)voice.next=new VoicePeer {epoch=epoch,next=voice.next};
            voice=voice.next;
        }
        return null;
    }
    void AddVoicePacket(VoicePeer voice,long sample,double time,byte[] data)
    {
        long late=voice.decoder.Late;bool accepted=voice.decoder.Add(sample,data);LatePackets+=voice.decoder.Late-late;
        if(!accepted)return;
        ReceivedPackets++;
        if(sample>voice.lastPacket)
        {
            voice.clock.Observe(sample,time);voice.lastPacket=sample;
        }
    }
    void StartVoice(int slot,int epoch,double time,long firstSample,int delay)
    {
        var voice=GetVoice(slot,epoch);if(voice==null || voice.decoder!=null)return;
        voice.time=time;voice.delay=delay;voice.clock.Observe(firstSample,time+(double)firstSample/VoiceCodec.Rate);
        voice.decoder=new VoiceCodec.Receiver(session.manager.sampleRate,(data,count,status)=>
        {
            var stream=Remote(slot).Stream;if(stream!=null && !stream.Finished)stream.Append(data,0,count,session.manager.LiveSample,status);
        },delay,firstSample,VoiceWindow);
        if(voice.final!=long.MaxValue)voice.decoder.Stop(voice.final);
        voice.pending.Sort((a,b)=>a.sample.CompareTo(b.sample));
        foreach(var frame in voice.pending)AddVoicePacket(voice,frame.sample,frame.time,frame.data);
        voice.pending.Clear();
    }
    void Update()
    {
        if(!Ready || replay || !session.manager.IsLiveSimulation || session.manager.IsSourceReplay)return;
        var manager=session.manager;long sample=manager.LiveSample;
        if(Connected)
        {
            poses[LocalSlot]=new Pose(session.player.position,manager.takeListeners[manager.monitorListener].rotation);poseEpochs[LocalSlot]=localEpoch;
            double now=SessionTime;poseTimes[LocalSlot]=now;
            if(now-lastSend>=1.0/30)
            {
                lastSend=now;
                if(IsHost)
                {
                    var frame=new PoseFrame {time=now,poses=new Pose[Capacity+sharedObjects.Length],times=new double[Capacity],present=(bool[])present.Clone(),epochs=(int[])poseEpochs.Clone()};
                    Array.Copy(poses,frame.poses,Capacity);Array.Copy(poseTimes,frame.times,Capacity);for(int i=0;i<sharedObjects.Length;i++)frame.poses[Capacity+i]=new Pose(sharedObjects[i].position,sharedObjects[i].rotation);
                    AddFrame(frame);
                    Broadcast(Kind.World,w=> {
                        w.Write(now);for(int i=0;i<Capacity;i++) {w.Write(frame.present[i]);w.Write(frame.epochs[i]);w.Write(frame.times[i]);WritePose(w,frame.poses[i]);}
                        for(int i=Capacity;i<frame.poses.Length;i++)WritePose(w,frame.poses[i]);
                    },false);
                }
                else Send(0,Kind.Pose,w=> {w.Write(localEpoch);w.Write(now);WritePose(w,poses[LocalSlot]);});
            }
            double presentation=now-presentationSeconds;
            if(frames.Count>0)
            {
                for(int i=0;i<Capacity+sharedObjects.Length;i++)
                {
                    double Stamp(PoseFrame f)=>i<Capacity?f.times[i]:f.time;
                    var previous=frames[0];var next=previous;
                    foreach(var frame in frames)
                    {
                        if(Stamp(frame)<=presentation)previous=next=frame;
                        else {next=frame;break;}
                    }
                    if(Stamp(previous)>presentation)continue;
                    double duration=Stamp(next)-Stamp(previous);
                    float fraction=duration<=0?0:(float)Math.Clamp((presentation-Stamp(previous))/duration,0,1);
                    var a=previous.poses[i];var b=next.poses[i];var p=new Pose(Vector3.Lerp(a.position,b.position,fraction),Quaternion.Slerp(a.rotation,b.rotation,fraction));
                    if(i<Capacity)
                    {
                        presented[i]=previous.epochs[i]==next.epochs[i]?p:a;
                        if(i!=LocalSlot && shownEpochs[i]!=previous.epochs[i])manager.MarkTakeDiscontinuity(Remote(i).transform.parent);
                        shownEpochs[i]=previous.epochs[i];
                        if(shown[i] && !previous.present[i])voices[i].decoder?.Stop(voices[i].decoder.Sample);
                        shown[i]=previous.present[i];
                    }
                    else if(!IsHost)sharedObjects[i-Capacity].SetPositionAndRotation(p.position,p.rotation);
                }
                Physics.SyncTransforms();
            }
        }

        for(int i=0;i<Capacity;i++)if(i!=LocalSlot)
        {
            var source=Remote(i);var root=source.transform.parent;
            bool visible=Connected && shown[i];
            if(visible && !root.gameObject.activeSelf)manager.MarkTakeDiscontinuity(root);
            root.gameObject.SetActive(visible);
            if(visible)
            {
                root.SetPositionAndRotation(presented[i].position,Quaternion.Euler(0,presented[i].rotation.eulerAngles.y,0));
                source.transform.rotation=presented[i].rotation;
            }
            PumpVoice(i,sample);
        }
    }
    void AddFrame(PoseFrame frame)
    {
        if(frames.Count>0 && frame.time<=frames[frames.Count-1].time)return;
        frames.Add(frame);
        // Retain the interpolation window and one second of jitter, independent of session duration.
        int limit=Math.Max(4,(int)Math.Ceiling((presentationSeconds+1)*30));
        if(frames.Count>limit)frames.RemoveAt(0);
    }
    void PumpVoice(int slot,long sample)
    {
        var voice=voices[slot];var source=Remote(slot);var stream=source.Stream;
        if(stream==null || stream.Finished)return;
        while(voice.next!=null && voice.decoder!=null && voice.decoder.Sample>=voice.decoder.Final)
        {
            voice.decoder.DecodeNext();voice.decoder.Dispose();voice=voices[slot]=voice.next;
        }
        // Payload may precede its reliable epoch metadata. Do not commit silence over it.
        if(voice.decoder==null) {if(voice.pending.Count==0)stream.SilenceUntil(sample,sample);return;}
        int rate=session.manager.sampleRate;
        double Scheduled()=>sample+(voice.clock.At(voice.decoder.Sample)+presentationSeconds+voice.extraDelay-SessionTime)*rate;
        if(voice.bound!=stream)
        {
            voice.bound=stream;
            long scheduled=(long)Math.Round(Scheduled());
            voice.firstSample=Math.Max(stream.End,scheduled<sample?sample+(long)Math.Ceiling(presentationSeconds*rate):scheduled);
            voice.extraDelay+=(double)(voice.firstSample-scheduled)/rate;
            session.manager.RecordStreamEpoch(source,new AcousticTake.StreamEpoch {sample=voice.firstSample,rate=48000,channels=1,participant=slot.ToString(),epoch=voice.epoch.ToString(),
                remoteTime=voice.clock.At(voice.decoder.Sample),uncertaintySeconds=-1,presentationSeconds=presentationSeconds+voice.extraDelay,
                processing="Opus mono 48k/20ms/32kbps; codec delay removed; PLC at deadline; sample-clock drift correction"});
        }
        stream.SilenceUntil(Math.Min(sample,voice.firstSample),sample);
        while(stream.End<=sample && sample>=voice.firstSample)
        {
            voice.decoder.CorrectClock((Scheduled()-stream.End-voice.decoder.PendingSamples)/rate);
            long concealed=voice.decoder.Concealed;bool decoded=voice.decoder.DecodeNext();ConcealedFrames+=voice.decoder.Concealed-concealed;
            if(!decoded) {if(voice.next==null)stream.SilenceUntil(sample,sample);break;}
        }
    }
    public double SceneSoundOffset=>Connected?Math.Max(0,SessionTime-hostOrigin):0;
    public void Shutdown()
    {
        closing=true;
        if(microphone!=null)microphone.StopCapture();
        if(net!=null) {net.Shutdown();Destroy(net.gameObject);net=null;}
        if(voices!=null)foreach(var first in voices)for(var voice=first;voice!=null;voice=voice.next)voice.decoder?.Dispose();
    }
    void OnDestroy()
    {
        if(microphone!=null) {microphone.VoiceState-=LocalVoiceState;microphone.VoiceSamples-=LocalVoiceSamples;}
        Shutdown();
    }
    public static string BuildFingerprint()
    {
#if UNITY_EDITOR
        string root=Path.GetFullPath(Path.Combine(Application.dataPath,".."));
        return SceneRecording.HashProject(root);
#else
        var stamp=Resources.Load<TextAsset>("NetworkBuildStamp");
        if(stamp==null)throw new InvalidOperationException("Build the player with the acoustic network build stamp.");return stamp.text.Trim();
#endif
    }
}
