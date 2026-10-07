using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Collections;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using UnityEngine;

// LAN transport adapter. Acoustic capture stays local to each client, after presentation/voice decoding.
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
    public int ReceivedPackets {get;private set;}
    public int ConcealedFrames {get;private set;}
    public int LatePackets {get;private set;}
    public int ConnectedPlayers=>IsHost?clients.Count:shown==null?0:shown.Count(value=>value);
    public int Capacity=>remoteVoices.Length+1;
    public static string PendingMode,PendingAddress;
    NetworkManager net;
    readonly Dictionary<ulong,int> clients=new Dictionary<ulong,int>();
    ulong[] occupants;
    bool[] present;
    Pose[] poses;
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
    const string Message="acoustics-v1";
    enum Kind : byte {Welcome,Pose,World,Interact,VoiceStart,VoicePacket,VoiceStop}
    sealed class PoseFrame {public double time;public Pose[] poses;public bool[] present;public int[] epochs;}
    sealed class VoicePeer
    {
        public int epoch;
        public double time;
        public long start,firstSample;
        public SourceStream bound;
        public VoiceCodec.Receiver decoder;
    }
    public void Prepare(AcousticTake.SessionDefinition saved=null)
    {
        replay=saved!=null;LocalSlot=saved?.localSlot??0;
        if(saved!=null && saved.slots!=Capacity)throw new InvalidDataException("Recorded roster does not match this scene.");
        occupants=new ulong[Capacity];present=new bool[Capacity];poses=new Pose[Capacity];voices=new VoicePeer[Capacity];presented=new Pose[Capacity];shown=new bool[Capacity];poseEpochs=new int[Capacity];shownEpochs=new int[Capacity];
        for(int i=0;i<Capacity;i++) {occupants[i]=ulong.MaxValue;poses[i]=new Pose(session.player.position,session.manager.takeListeners[session.manager.monitorListener].rotation);voices[i]=new VoicePeer();}
        foreach(var voice in remoteVoices)voice.transform.parent.gameObject.SetActive(false);
        microphone.source.mutedListenerPlusOne=microphone.monitorSelf?0:1;
        microphone.Clock=()=>Connected?net.LocalTime.Time:AudioSettings.dspTime;
        microphone.VoiceState+=LocalVoiceState;microphone.VoiceSamples+=LocalVoiceSamples;
        Ready=replay;Status=replay?"Replay (offline)":"Solo";
    }
    public IEnumerator Connect()
    {
        string mode=PendingMode;PendingMode=null;
        if(string.IsNullOrEmpty(mode)) {Ready=true;yield break;}
        if(PendingAddress!=null)address=PendingAddress;PendingAddress=null;
        Status="Connecting";
        try
        {
            if(spawnPoints.Length!=Capacity || Array.Exists(spawnPoints,p=>p==null))throw new InvalidOperationException("Assign one spawn point per player slot.");
            var go=new GameObject("LAN transport");net=go.AddComponent<NetworkManager>();
            var transport=go.AddComponent<UnityTransport>();transport.SetConnectionData(address,port,"0.0.0.0");
            net.NetworkConfig=new NetworkConfig {NetworkTransport=transport,EnableSceneManagement=false,ConnectionApproval=true,
                ProtocolVersion=3,TickRate=30,ForceSamePrefabs=false};
            fingerprint=BuildFingerprint()+"|"+gameObject.scene.path+"|"+Capacity;
            net.NetworkConfig.ConnectionData=Encoding.UTF8.GetBytes(fingerprint);
            net.ConnectionApprovalCallback=Approve;
            net.OnClientConnectedCallback+=Joined;net.OnClientDisconnectCallback+=Left;
            if(!(mode=="host"?net.StartHost():net.StartClient()))throw new InvalidOperationException("Could not start LAN transport.");
            net.CustomMessagingManager.RegisterNamedMessageHandler(Message,Receive);
        }
        catch(Exception e) {Status=e.Message;Shutdown();yield break;}
        double deadline=Time.realtimeSinceStartupAsDouble+15;
        while(!Ready && net!=null && Time.realtimeSinceStartupAsDouble<deadline)yield return null;
        if(!Ready) {Status="Connection failed: "+(net?.DisconnectReason??"timeout");Shutdown();yield break;}
        session.player.SetPositionAndRotation(spawnPoints[LocalSlot].position,spawnPoints[LocalSlot].rotation);Physics.SyncTransforms();
        Status=IsHost?"Hosting :"+port:"Connected to "+address;
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
        if(id==net.LocalClientId) {sessionId=Guid.NewGuid().ToString("N");hostOrigin=net.ServerTime.Time;LocalSlot=slot;Ready=true;}
        else
        {
            Send(id,Kind.Welcome,w=> {w.Write(slot);w.Write(sessionId);w.Write(hostOrigin);},true);
            for(int source=0;source<Capacity;source++)
            {
                int current=source;var voice=voices[source];
                if(source==LocalSlot && encoder!=null)
                    Send(id,Kind.VoiceStart,w=> {w.Write(current);w.Write(voiceEpoch);w.Write(voiceStart);w.Write(encoder.Frames);},true);
                else if(voice.decoder!=null && voice.decoder.Next<voice.decoder.Final)
                    Send(id,Kind.VoiceStart,w=> {w.Write(current);w.Write(voice.epoch);w.Write(voice.time);w.Write(voice.decoder.Next);},true);
            }
        }
    }
    void Left(ulong id)
    {
        if(closing)return;
        if(IsHost && clients.TryGetValue(id,out int slot))
        {
            present[slot]=false;voices[slot].decoder?.Stop(voices[slot].decoder.Next);clients.Remove(id);
            // Do not reuse a participant slot inside a recording: delayed packets/tails retain their identity.
        }
        if(net!=null && id==net.LocalClientId) {Status="Disconnected: "+net.DisconnectReason;session.SaveRecording();}
    }
    public AcousticTake.SessionDefinition Description()=>new AcousticTake.SessionDefinition {scene=gameObject.scene.path,
        sessionId=sessionId??Guid.NewGuid().ToString("N"),build=fingerprint,clientId=Connected?net.LocalClientId.ToString():"local",mode=Connected?"lan-client-presented-v1":"solo",
        slots=Capacity,localSlot=LocalSlot,serverTimeOrigin=Connected?net.ServerTime.Time:0};
    public void CaptureStarted()
    {
        if(session.manager.SessionInfo!=null)session.manager.SessionInfo.serverTimeOrigin=
            (Connected?net.ServerTime.Time:AudioSettings.dspTime)-(double)session.manager.LiveSample/session.manager.sampleRate;
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
                poseEpochs[slot]=reader.ReadInt32();poses[slot]=ReadPose(reader);
            }
            else if(kind==Kind.World)
            {
                if(IsHost || sender!=0)return;
                double time=reader.ReadDouble();if(!double.IsFinite(time))throw new InvalidDataException("Invalid network time.");
                var frame=new PoseFrame {time=time,poses=new Pose[Capacity+sharedObjects.Length],present=new bool[Capacity],epochs=new int[Capacity]};
                for(int i=0;i<Capacity;i++) {frame.present[i]=reader.ReadBoolean();frame.epochs[i]=reader.ReadInt32();frame.poses[i]=ReadPose(reader);}
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
                    StartVoice(slot,epoch,time,reader.ReadInt32());
                }
                else if(kind==Kind.VoicePacket)
                {
                    int sequence=reader.ReadInt32(),count=reader.ReadInt32();
                    if(count<1 || count>VoiceCodec.MaximumPacket || count!=reader.BaseStream.Length-reader.BaseStream.Position)throw new InvalidDataException("Invalid voice packet.");
                    var data=reader.ReadBytes(count);var voice=voices[slot];
                    if(voice.epoch==epoch && voice.decoder!=null) {voice.decoder.Add(sequence,data);ReceivedPackets++;}
                }
                else {int frames=reader.ReadInt32();if(voices[slot].epoch==epoch)voices[slot].decoder?.Stop(frames);}
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
        net.CustomMessagingManager.SendNamedMessage(Message,client,writer,reliable?NetworkDelivery.ReliableFragmentedSequenced:NetworkDelivery.UnreliableSequenced);
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
            voiceEpoch++;voiceStart=time;encoder=new VoiceCodec.Encoder((sequence,data,count)=>Broadcast(Kind.VoicePacket,w=>
                {w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(sequence);w.Write(count);w.Write(data,0,count);},false));
            Broadcast(Kind.VoiceStart,w=> {w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(time);w.Write(0);},true);
        }
        else if(encoder!=null)
        {
            encoder.Finish();int frames=encoder.Frames;encoder=null;
            Broadcast(Kind.VoiceStop,w=> {w.Write(LocalSlot);w.Write(voiceEpoch);w.Write(frames);},true);
        }
    }
    void LocalVoiceSamples(float[] data,int count)=>encoder?.Append(data,count);
    void StartVoice(int slot,int epoch,double time,int firstFrame)
    {
        var voice=voices[slot];if(epoch<=voice.epoch)return;
        voice.decoder?.Dispose();voice.epoch=epoch;voice.time=time;voice.bound=null;
        voice.decoder=new VoiceCodec.Receiver(AudioSettings.outputSampleRate,(data,count,status)=>
        {
            var stream=Remote(slot).Stream;if(stream!=null && !stream.Finished)stream.Append(data,0,count,session.manager.LiveSample,status);
        },firstFrame,(int)Math.Ceiling((presentationSeconds+1)*50));
    }
    void Update()
    {
        if(!Ready || replay || !session.manager.IsLiveSimulation || session.manager.IsSourceReplay)return;
        var manager=session.manager;long sample=manager.LiveSample;
        if(Connected)
        {
            poses[LocalSlot]=new Pose(session.player.position,manager.takeListeners[manager.monitorListener].rotation);poseEpochs[LocalSlot]=localEpoch;
            double now=net.ServerTime.Time;
            if(now-lastSend>=1.0/30)
            {
                lastSend=now;
                if(IsHost)
                {
                    var frame=new PoseFrame {time=now,poses=new Pose[Capacity+sharedObjects.Length],present=(bool[])present.Clone(),epochs=(int[])poseEpochs.Clone()};
                    Array.Copy(poses,frame.poses,Capacity);for(int i=0;i<sharedObjects.Length;i++)frame.poses[Capacity+i]=new Pose(sharedObjects[i].position,sharedObjects[i].rotation);
                    AddFrame(frame);
                    Broadcast(Kind.World,w=> {
                        w.Write(now);for(int i=0;i<Capacity;i++) {w.Write(frame.present[i]);w.Write(frame.epochs[i]);WritePose(w,frame.poses[i]);}
                        for(int i=Capacity;i<frame.poses.Length;i++)WritePose(w,frame.poses[i]);
                    },false);
                }
                else Send(0,Kind.Pose,w=> {w.Write(localEpoch);WritePose(w,poses[LocalSlot]);});
            }
            double presentation=now-presentationSeconds;
            while(frames.Count>1 && frames[1].time<=presentation)frames.RemoveAt(0);
            if(frames.Count>0 && frames[0].time<=presentation)
            {
                var previous=frames[0];var next=frames.Count>1?frames[1]:previous;
                float fraction=next==previous?0:(float)((presentation-previous.time)/(next.time-previous.time));
                for(int i=0;i<next.poses.Length;i++)
                {
                    var a=previous.poses[i];var b=next.poses[i];var p=new Pose(Vector3.Lerp(a.position,b.position,fraction),Quaternion.Slerp(a.rotation,b.rotation,fraction));
                    if(i<Capacity)
                    {
                        presented[i]=previous.epochs[i]==next.epochs[i]?p:a;
                        if(i!=LocalSlot && shownEpochs[i]!=previous.epochs[i])manager.MarkTakeDiscontinuity(Remote(i).transform.parent);
                        shownEpochs[i]=previous.epochs[i];
                        if(shown[i] && !previous.present[i])voices[i].decoder?.Stop(voices[i].decoder.Next);
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
        if(voice.decoder==null) {stream.SilenceUntil(sample,sample);return;}
        if(voice.bound!=stream)
        {
            bool continuation=voice.bound!=null;voice.bound=stream;
            long scheduled=sample+(long)Math.Round((voice.time+presentationSeconds-net.ServerTime.Time)*session.manager.sampleRate);
            long offset=(long)voice.decoder.Next*session.manager.sampleRate/50;
            // A late control message still needs a prebuffer. Starting at the already
            // finalized cursor would conceal every frame before its packet could arrive.
            long buffered=sample+(long)Math.Ceiling(presentationSeconds*session.manager.sampleRate)-offset;
            voice.start=continuation?sample-offset:Math.Max(buffered,scheduled);voice.firstSample=voice.start+offset;
            session.manager.RecordStreamEpoch(source,new AcousticTake.StreamEpoch {sample=voice.start+offset,rate=48000,channels=1,participant=slot.ToString(),epoch=voice.epoch.ToString(),
                remoteTime=voice.time+voice.decoder.Next/50.0,uncertaintySeconds=-1,presentationSeconds=presentationSeconds+(double)(voice.start-scheduled)/session.manager.sampleRate,processing="Opus mono 48k/20ms/32kbps; PLC at deadline; Speex quality 8; client presentation"});
        }
        stream.SilenceUntil(Math.Min(sample,voice.firstSample),sample);
        while(voice.start+(long)voice.decoder.Next*session.manager.sampleRate/50<=sample)
        {
            if(!voice.decoder.DecodeNext()) {stream.SilenceUntil(sample,sample);break;}
        }
        ConcealedFrames=LatePackets=0;
        foreach(var v in voices)if(v.decoder!=null) {ConcealedFrames+=v.decoder.Concealed;LatePackets+=v.decoder.Late;}
    }
    public double SceneSoundOffset=>Connected?Math.Max(0,net.ServerTime.Time-hostOrigin):0;
    public void Shutdown()
    {
        closing=true;
        if(microphone!=null)microphone.StopCapture();
        if(net!=null) {net.Shutdown();Destroy(net.gameObject);net=null;}
        if(voices!=null)foreach(var voice in voices)voice.decoder?.Dispose();
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
