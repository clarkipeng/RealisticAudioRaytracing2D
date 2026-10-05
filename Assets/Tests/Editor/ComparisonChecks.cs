using System;
using System.IO;
using UnityEngine;

public static partial class TakeChecks
{
    static void ComparisonInspection()
    {
        var real=new double[64]; var imaginary=new double[64]; var original=new double[64];
        for(int n=0;n<64;n++) real[n]=original[n]=Math.Sin(n*.37)+Math.Cos(n*.81)*.3;
        TakeComparisonAudio.Fft(real,imaginary); double fftError=0;
        for(int k=0;k<64;k++)
        {
            double r=0,i=0;
            for(int n=0;n<64;n++) {double a=-2*Math.PI*k*n/64; r+=original[n]*Math.Cos(a); i+=original[n]*Math.Sin(a);}
            fftError=Math.Max(fftError,Math.Max(Math.Abs(real[k]-r),Math.Abs(imaginary[k]-i)));
        }
        Check(fftError<1e-10,"Display FFT agrees with independent direct DFT, including phase.");
        var tone=new float[512*2]; for(int n=0;n<512;n++) {tone[n*2]=(float)(.25*Math.Sin(2*Math.PI*8*n/64)); tone[n*2+1]=0;}
        var spectrum=TakeComparisonAudio.Spectrogram(tone,2,0,64,16);
        var silent=TakeComparisonAudio.Spectrogram(tone,2,1,64,16);
        Near(spectrum.Db[8*spectrum.Width],(float)(20*Math.Log10(.25)),"Spectrogram uses calibrated peak-amplitude dBFS",1e-4f);
        Check(silent.Db[0]==-240 && spectrum.Hop==16,"Channels stay separate and requested small-window hop is retained.");
        var dense=TakeComparisonAudio.Spectrogram(new float[100000],1,0,64,16);
        Check(dense.Width<=512 && dense.Hop%16==0,"Spectrogram display decimation has an explicit hop and bounded columns.");
        Reject(()=>TakeComparisonAudio.Spectrogram(tone,2,0,65,16),"Reject unsupported FFT without silently changing resolution.");
        var env=TakeComparisonAudio.Envelope(new[]{0f,1f,-2f,.3f,0f,.1f},1,0,2);
        Check(env[0].x==-2 && env[0].y==1,"Waveform envelope retains extrema rather than missing a narrow peak.");
        var low=new float[800]; var high=new float[800];
        for(int n=0;n<400;n++) {low[n*2]=1; low[n*2+1]=2; high[n*2]=.5f; high[n*2+1]=1;}
        var audition=TakeComparisonAudio.Audition(low,high,2,100,2,.5f,1000);
        Check(audition.Length==600 && audition[20]==.25f && audition[21]==.5f,"Difference audition preserves ear order, cursor and shared gain.");
        Check(audition[0]==0 && audition[audition.Length-1]==0 && low[0]==1,"Audition fade does not mutate source samples.");
        string wav=Path.Combine(directory,"inspection.wav");
        using(var writer=new BinaryWriter(File.Create(wav)))
        {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36+low.Length*4);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16); writer.Write((ushort)3); writer.Write((ushort)2);
            writer.Write(1000); writer.Write(8000); writer.Write((ushort)8); writer.Write((ushort)32);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(low.Length*4); foreach(float value in low) writer.Write(value);
        }
        var run=new TakeRender.Report {sampleRate=1000,outputSamples=400,channelsPerListener=2};
        using(var wave=new TakeComparisonAudio.Wave(wav,AcousticTake.HashFile(wav),run))
        {var part=wave.Read(123,7); Check(part.Length==14 && part[0]==1 && part[1]==2,"WAV reader seeks in sample frames."); Reject(()=>wave.Read(398,3),"Reject out-of-range preview.");}
        Reject(()=>new TakeComparisonAudio.Wave(wav,"wrong-hash",run),"Reject changed audio before audition.");
        run.sampleRate=2000; Reject(()=>new TakeComparisonAudio.Wave(wav,AcousticTake.HashFile(wav),run),"Reject WAV/report rate mismatch.");
        using(var take=AcousticTake.Open(Path.Combine(directory,"binaural-cpu")))
        {
            var settings=new TakeRender.Settings {listenerOutput=TakeRender.ListenerOutput.Binaural,responseStepSamples=128,transitionSamples=32,tailSamples=64};
            var fine=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(settings)); fine.responseStepSamples=64;
            string output=Path.Combine(directory,"inspect-comparison"),path=Path.Combine(output,"comparison.json");
            var result=TakeComparison.Run(take,settings,fine,output);
            var loaded=TakeComparisonAudio.OpenReport(path);
            Check(loaded.listenerIds.Length==2 && loaded.takeDirectory==Path.GetRelativePath(output,take.DirectoryPath),"Comparison retains listener identity and portable take location.");
            using(var wave=new TakeComparisonAudio.Wave(Path.Combine(output,"high","listener-1.wav"),loaded.high.audioHashes[1],loaded.high))
                Check(wave.Channels==2 && wave.Frames==loaded.high.outputSamples,"Generated stereo comparison opens without scene dependencies.");
            loaded.high.hrtfHash="different"; File.WriteAllText(path,JsonUtility.ToJson(loaded)); Reject(()=>TakeComparisonAudio.OpenReport(path),"Reject incompatible ear rendering.");
            result.complete=false; File.WriteAllText(path,JsonUtility.ToJson(result)); Reject(()=>TakeComparisonAudio.OpenReport(path),"Reject incomplete comparison.");
        }
    }
    static void ComparisonViews()
    {
        var fixtureScene=UnityEditor.SceneManagement.EditorSceneManager.NewScene(UnityEditor.SceneManagement.NewSceneSetup.EmptyScene,UnityEditor.SceneManagement.NewSceneMode.Additive);
        var root=new GameObject("Comparison view fixture");
        var source=GameObject.CreatePrimitive(PrimitiveType.Cube); source.transform.SetParent(root.transform); source.transform.position=new Vector3(1,0,4);
        var listener=new GameObject("Fixture listener"); listener.transform.SetParent(root.transform); listener.transform.position=new Vector3(0,0,-3);
        var floor=GameObject.CreatePrimitive(PrimitiveType.Cube); floor.transform.SetParent(root.transform);
        floor.transform.position=new Vector3(0,-1,0); floor.transform.localScale=new Vector3(20,1,20);
        var manager=root.AddComponent<RayTraceManager>(); var scene=Scene();
        manager.takeObjects=new System.Collections.Generic.List<RayTraceManager.TakeObjectBinding> {
            new RayTraceManager.TakeObjectBinding {id=scene.objects[0].id,transform=source.transform},
            new RayTraceManager.TakeObjectBinding {id=scene.objects[1].id,transform=listener.transform}};
        string path=Path.Combine(directory,"view-take");
        var state=States(); state[0].position=Vector3.forward*4; state[1].position=Vector3.zero;
        var originalSource=source.transform.position; var originalListener=listener.transform.position;
        try
        {
            using(var writer=new AcousticTake.Writer(path,1000,scene,new AcousticTake.Primitive[0],new[]{new float[10]}))
            {writer.Append(0,state); state[0].position=Vector3.right*2+Vector3.forward*4; state[1].position=Vector3.right; state[1].rotation=Quaternion.Euler(0,90,0); writer.Append(1000,state); writer.Complete();}
            TakeComparison.Report report;
            using(var take=AcousticTake.Open(path)) report=new TakeComparison.Report {sceneHash=take.Metadata.sceneHash,stateHash=take.Metadata.stateHash,eventHash=take.Metadata.eventHash};
            using(var view=new TakeComparisonView(path,report))
            {
                view.SetSample(500,0);
                Near(Vector3.Distance(view.ObjectPosition(0),new Vector3(1,0,4)),0,"Visual replay interpolates captured source position",1e-6f);
                Near(Vector3.Distance(view.ListenerCamera.transform.position,Vector3.right*.5f),0,"Listener view shares the recorded sample",1e-6f);
                Near(Quaternion.Angle(view.ListenerCamera.transform.rotation,Quaternion.Euler(0,45,0)),0,"Listener view follows head orientation",1e-4f);
                Check(UnityEditor.SceneManagement.EditorSceneManager.IsPreviewScene(view.ListenerCamera.scene),"Views use an isolated preview scene.");
                int audio=0,scripts=0;
                foreach(var visual in view.ListenerCamera.scene.GetRootGameObjects()) {audio+=visual.GetComponentsInChildren<AudioListener>(true).Length+visual.GetComponentsInChildren<AudioSource>(true).Length; scripts+=visual.GetComponentsInChildren<MonoBehaviour>(true).Length;}
                Check(audio==0 && scripts==0,"Visual replay contains no gameplay or audio components.");
                try
                {
                    var rect=new Rect(0,0,100,100);
                    view.Input(new Event {type=EventType.MouseDown,button=1,mousePosition=new Vector2(20,20)},rect,new Rect(200,0,100,100));
                    view.Input(new Event {type=EventType.KeyDown,keyCode=KeyCode.W},rect,new Rect(200,0,100,100)); view.Tick();
                    view.MoveCamera(Vector3.down*20);
                    float floorY=view.ListenerCamera.transform.position.y;
                    Check(floorY>-.4f && floorY<-.3f,"Inspection camera sweeps against the ground even for a large movement.");
                    view.MoveCamera(new Vector3(1,-1,0));
                    Check(view.ListenerCamera.transform.position.x>1 && view.ListenerCamera.transform.position.y>-.4f,"Inspection camera slides along the ground instead of sticking or tunnelling.");
                    var freePosition=view.ListenerCamera.transform.position; view.SetSample(800,0);
                    Check(view.FreeCamera && view.ListenerCamera.transform.position==freePosition,"Free camera stays independent of recorded receiver motion.");
                    view.FollowListener(); Near(Vector3.Distance(view.ListenerCamera.transform.position,Vector3.right*.8f),0,"Follow restores recorded camera after free inspection",1e-6f);
                }
                finally {view.EndDrag();}
                view.SetSample(2000,0); Check(view.Sample==1000,"Visual replay holds its final pose through audio tails.");
                Check(source.transform.position==originalSource && listener.transform.position==originalListener,"Viewing a take leaves authoring transforms untouched.");
            }
        }
        finally {UnityEditor.SceneManagement.EditorSceneManager.CloseScene(fixtureScene,true);}
    }

}
