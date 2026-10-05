using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public static partial class TakeChecks
{
    static void OfflineBatchChecks(ComputeShader shader)
    {
        string input=Path.Combine(directory,"offline-batch-input");var scene=Scene(2);var state=States(2);
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="opaque",frequencies=new float[0],absorption=new float[0],scattering=new float[0]}};
        state[0].flags|=AcousticTake.StateFlags.Geometry;
        var geometry=new[]{new AcousticTake.Primitive {a=new Vector3(.5f,-1,-1),b=new Vector3(.5f,1,-1),c=new Vector3(.5f,0,1)}};
        var pcm=new float[113];for(int i=0;i<pcm.Length;i++)pcm[i]=Mathf.Sin(i*.13f)*.1f;
        using(var writer=new AcousticTake.Writer(input,8000,scene,geometry,new[]{pcm}))
        {
            for(int f=0;f<8;f++)
            {
                state[0].rotation=Quaternion.Euler(0,f*37,f*13);state[0].position=new Vector3(f*.17f,f*.04f,0);
                state[1].position=new Vector3(1,f*.1f,0);state[2].position=new Vector3(-1,f*.2f,0);
                state[1].flags=AcousticTake.StateFlags.Listener|AcousticTake.StateFlags.Active|(f==4?AcousticTake.StateFlags.Cut:0);
                writer.Append(f*167,state);
            }
            writer.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
            writer.Append(new AcousticTake.VoiceEvent {sample=300,action=AcousticTake.VoiceAction.Gain,gain=.5f});
            writer.Append(new AcousticTake.VoiceEvent {sample=1100,action=AcousticTake.VoiceAction.Stop});writer.Complete();
        }
        using var take=AcousticTake.Open(input);
        foreach(bool bvh in new[]{false,true})foreach(var filter in new[]{TakeRender.DelayFilter.Linear,TakeRender.DelayFilter.WindowedSinc})
        {
            TakeRender.Report reference=null;
            foreach(int capacity in new[]{1,3,8,64})
            {
                var settings=new TakeRender.Settings {useBvh=bvh,responseStepSamples=31,maximumResponsesPerBatch=capacity,
                    listenerOutput=TakeRender.ListenerOutput.Binaural,delayFilter=filter,sincRadius=2,tailSamples=53,transitionSamples=4};
                var result=TakeRender.RenderDirect(take,settings,Path.Combine(directory,"batch-"+bvh+"-"+filter+"-"+capacity),null,shader);
                if(reference==null)reference=result;
                else
                {
                    Check(result.responseHash==reference.responseHash,"Offline batch sizes preserve response bytes, including the final partial batch.");
                    for(int listener=0;listener<2;listener++)Check(result.audioHashes[listener]==reference.audioHashes[listener],"Offline batching preserves moving stereo output, cuts, events and tails.");
                }
            }
        }
    }
    static void BvhChecks(ComputeShader shader=null)
    {
        StaticBvhChecks(shader);
        var random=new System.Random(62831);
        float Coordinate() => (float)(random.NextDouble()*12-6);
        Vector3 Point() => new Vector3(Coordinate(),Coordinate(),Coordinate());
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        foreach(int count in new[]{0,1,4,5,257})
        {
            var scene=new AcousticTake.Scene {dimension=dimension,objects=new AcousticTake.ObjectDefinition[8],
                sources=new[]{new AcousticTake.SourceDefinition {objectIndex=4},new AcousticTake.SourceDefinition {objectIndex=5}},listeners=new[]{6,7}};
            var geometry=new AcousticTake.Primitive[count];
            for(int i=0;i<count;i++)
            {
                var center=Point();geometry[i]=new AcousticTake.Primitive {objectIndex=i%4,a=center,b=center+Point()*.4f,c=center+Point()*.4f};
            }
            if(count>0)geometry[0]=new AcousticTake.Primitive {objectIndex=0,a=new Vector3(0,-1,-1),b=new Vector3(0,1,-1),c=new Vector3(0,0,1)};
            var state=new AcousticTake.State[8];var expected=new TakeRender.DirectPath[4];var actual=new TakeRender.DirectPath[4];
            var settings=new TakeRender.Settings {useBvh=true};
            using var oracle=new DirectSolver(scene,geometry,new TakeRender.Settings(),shader,3);
            using var accelerated=new DirectSolver(scene,geometry,settings,shader,3);
            var batch=new AcousticTake.State[24];var expectedBatch=new TakeRender.DirectPath[12];var actualBatch=new TakeRender.DirectPath[12];
            for(int frame=0;frame<12;frame++)
            {
                for(int i=0;i<state.Length;i++)state[i]=new AcousticTake.State {position=i<2?Vector3.zero:i<4?new Vector3(frame*.13f*i,0,0):Point(),
                    rotation=i<2?Quaternion.identity:Quaternion.Euler(dimension==AcousticTake.Dimension.Planar2D?0:frame*3,0,frame*11),
                    flags=AcousticTake.StateFlags.Active|(i<4?AcousticTake.StateFlags.Geometry:i<6?AcousticTake.StateFlags.Source:AcousticTake.StateFlags.Listener)};
                state[4].position=new Vector3(-2,0,0);state[6].position=frame==0?state[4].position:new Vector3(2,0,0);
                if(frame==2)state[0].flags&=~AcousticTake.StateFlags.Active;
                if(frame==3)state[1].flags&=~AcousticTake.StateFlags.Geometry;
                oracle.Solve(state,expected);accelerated.Solve(state,actual);
                for(int p=0;p<actual.Length;p++)Check(actual[p].Equals(expected[p]),"BVH preserves brute-force response ("+dimension+", "+count+", frame "+frame+").");
                Array.Copy(state,0,batch,(frame%3)*8,8);
                if(frame%3==2)
                {
                    oracle.BeginSolve(batch,3);accelerated.BeginSolve(batch,3);
                    oracle.TryComplete(expectedBatch,true);accelerated.TryComplete(actualBatch,true);
                    for(int p=0;p<actualBatch.Length;p++)Check(actualBatch[p].Equals(expectedBatch[p]),"BVH batches share static bounds and test moving geometry at each frame's pose.");
                }
            }
            if(shader==null)
            {
                accelerated.Solve(state,actual);int updates=accelerated.BvhUpdates;
                long before=GC.GetAllocatedBytesForCurrentThread();
                for(int i=0;i<10;i++)accelerated.Solve(state,actual);
                Check(GC.GetAllocatedBytesForCurrentThread()==before,"Warmed BVH solves allocate no managed memory.");
                Check(accelerated.BvhUpdates==updates,"Unchanged geometry does not rebuild when only a query repeats.");
                state[4].position+=Vector3.one;accelerated.Solve(state,actual);
                Check(accelerated.BvhUpdates==updates,"Moving endpoints do not rebuild geometry.");
            }
        }
    }
    static void StaticBvhChecks(ComputeShader shader)
    {
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        {
            var scene=new AcousticTake.Scene {dimension=dimension,objects=new AcousticTake.ObjectDefinition[4],
                sources=new[]{new AcousticTake.SourceDefinition {objectIndex=2}},listeners=new[]{3}};
            var geometry=new AcousticTake.Primitive[2];
            for(int i=0;i<2;i++)geometry[i]=new AcousticTake.Primitive {objectIndex=i,a=new Vector3(0,-1,-1),b=new Vector3(0,1,-1),c=new Vector3(0,0,1)};
            var state=new AcousticTake.State[4];
            for(int i=0;i<4;i++)state[i]=new AcousticTake.State {rotation=Quaternion.identity,flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry};
            state[0].position=new Vector3(0,10,0);state[2].position=new Vector3(-2,0,0);state[3].position=new Vector3(2,0,0);
            using var oracle=new DirectSolver(scene,geometry,new TakeRender.Settings(),shader,3);
            using var accelerated=new DirectSolver(scene,geometry,new TakeRender.Settings {useBvh=true},shader,3);
            var batch=new AcousticTake.State[12];var expected=new TakeRender.DirectPath[3];var actual=new TakeRender.DirectPath[3];
            for(int f=0;f<3;f++)
            {
                // The object starts moving inside the first batch, then returns to its initial pose.
                state[1].position=new Vector3(0,f==1?4:0,0);Array.Copy(state,0,batch,f*4,4);
            }
            oracle.BeginSolve(batch,3);accelerated.BeginSolve(batch,3);oracle.TryComplete(expected,true);accelerated.TryComplete(actual,true);
            for(int f=0;f<3;f++)Check(actual[f].Equals(expected[f]),"First motion within a batch never uses stale static bounds.");
            Check(actual[0].pressureGain==0 && actual[1].pressureGain>0 && actual[2].pressureGain==0,"Moving occluder changes visibility and remains correct when it returns.");
            Check(accelerated.StaticPrimitiveCount==1,"Only the fixed object's primitive remains in the BVH.");
            int builds=accelerated.BvhUpdates;
            expected=new TakeRender.DirectPath[1];actual=new TakeRender.DirectPath[1];
            for(int f=0;f<6;f++)
            {
                state[1].position=new Vector3(0,f%2==0?4:0,0);
                oracle.Solve(state,expected);accelerated.Solve(state,actual);
                Check(actual[0].Equals(expected[0]),"Ongoing motion uses the current pose outside the BVH.");
            }
            Check(accelerated.BvhUpdates==builds,"Moving and stopping known dynamic geometry never rebuilds the static tree.");
            state[1].flags&=~AcousticTake.StateFlags.Active;
            state[2].position+=Vector3.up*10;state[3].position+=Vector3.up*10;
            oracle.Solve(state,expected);accelerated.Solve(state,actual);
            Check(actual[0].Equals(expected[0]) && actual[0].pressureGain==0,"The retained static wall still occludes moving endpoints.");
            state[0].flags&=~AcousticTake.StateFlags.Geometry;
            oracle.Solve(state,expected);accelerated.Solve(state,actual);
            Check(actual[0].Equals(expected[0]) && actual[0].pressureGain>0 && accelerated.BvhUpdates==builds,"Static visibility flags change without rebuilding bounds.");
            state[0].flags|=AcousticTake.StateFlags.Geometry;state[0].position+=Vector3.up*10;
            oracle.Solve(state,expected);accelerated.Solve(state,actual);
            Check(actual[0].Equals(expected[0]) && accelerated.StaticPrimitiveCount==0,"Late movement of the last static object leaves a valid empty BVH.");
        }
    }
}
