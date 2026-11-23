using System;
using UnityEngine;

// Room modes on the GPU path reverb (RoomModes.cs; PathReverb.AddModes is the CPU reference): per update the CPU probes
// each pair's modal amplitudes (and decides where the bank applies); the GPU synthesises dh = W_m (h_modal - h_direct)
// at the decimated rate, transforms, crosses over, transforms back and adds it to the filtered partitions (the
// RoomMode kernels of GpuAudio.compute), and the path kernels scale the histogram by g_b (RoomModeShare).
internal sealed partial class GpuAudio
{
    int modeSynthesis,modeFft,modeCross,modeAdd;
    RoomModes modes;
    ComputeBuffer modeShareBuffer,modeAmplitudeBuffer,modePairBuffer,modeFirst,modeSecond,modeFinal;
    float[] modeShares,modeAmplitudes;
    uint[] modePairs;
    int modeSlots;   // per-update tables, one slot per response of a batch (as the head poses: AudioPoseOffset)
    double[] modeScratch;
    internal RoomModes Modes => modes;
    internal int ModalPairs;   // diagnostics: pairs given the modes by the latest response
    internal int ModeStages;   // checks only: 1 stops after the synthesis (ReadModes then reads h_modal x N x output rate)

    void PrepareModes(AcousticTake.Scene scene,int rate)
    {
        modes=RoomModes.ForScene(scene,settings,rate,checked(lateCount*latePartition));
        modeSlots=modes==null?1:headsPose.count/poses.Length;
        modeShares=new float[pairs*TakeRender.Settings.LateBands];Array.Fill(modeShares,1f);
        modeShareBuffer=Buffer("RoomModeShare",checked(modeSlots*modeShares.Length),4);
        for(int slot=0;slot<modeSlots;slot++)modeShareBuffer.SetData(modeShares,0,slot*modeShares.Length,modeShares.Length);
        if(modes==null){Buffer("RoomModeTable",1,16);Buffer("RoomModePhases",1,4);Buffer("RoomModeAmplitudes",1,4);Buffer("RoomModePairs",1,16);Buffer("RoomModeWeights",1,4);Buffer("RoomModeInterpolation",1,4);return;}
        int N=modes.Bank.Modes;
        Buffer("RoomModeTable",N,16).SetData(modes.ModeTable());
        Buffer("RoomModePhases",N,4).SetData(modes.ModePhases());
        var weights=new float[modes.Weights.Length];for(int k=0;k<weights.Length;k++)weights[k]=(float)modes.Weights[k];
        Buffer("RoomModeWeights",weights.Length,4).SetData(weights);
        Buffer("RoomModeInterpolation",modes.Interpolation.Length,4).SetData(Array.ConvertAll(modes.Interpolation,v=>(float)v));
        modeAmplitudes=new float[checked(pairs*N)];modePairs=new uint[4*pairs];modeScratch=new double[N];
        modeAmplitudeBuffer=Buffer("RoomModeAmplitudes",checked(modeSlots*modeAmplitudes.Length),4);
        modePairBuffer=Buffer("RoomModePairs",modeSlots*pairs,16);
        modeFirst=new ComputeBuffer(checked(pairs*modes.FftSize),8);modeSecond=new ComputeBuffer(checked(pairs*modes.FftSize),8);
        owned.Add(modeFirst);owned.Add(modeSecond);NumericBytes+=2L*pairs*modes.FftSize*8;
    }

    // The update's modal part: amplitudes, shares and the dh synthesis up to the inverse transform. True if any pair has it.
    bool SetModes(AcousticTake.Scene scene,AcousticTake.State[] states,bool valid,bool build)
    {
        if(modes==null)return false;
        int slot=poseCursor-1;Set("RoomModeOffset",slot);
        bool room=valid && build && modes.Applies(states);int N=modes.Bank.Modes;ModalPairs=0;
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners,source=pair/listeners;double direct=0,delay=0;
            bool modal=room && scene.sources[source].mutedListenerPlusOne!=listener+1 && TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener) &&
                modes.Pair(states[scene.sources[source].objectIndex].position,states[scene.listeners[listener]].position,settings.minimumDistance,modeScratch,out direct,out delay);
            for(int b=0;b<TakeRender.Settings.LateBands;b++)modeShares[pair*TakeRender.Settings.LateBands+b]=modal?(float)modes.Share[b]:1;
            modePairs[4*pair]=modal?1u:0;
            modePairs[4*pair+1]=modal?(uint)BitConverter.SingleToInt32Bits((float)(direct*modes.Rate)):0;
            modePairs[4*pair+2]=modal?modes.DirectPhase(delay):0;
            if(modal){for(int n=0;n<N;n++)modeAmplitudes[pair*N+n]=(float)modeScratch[n];ModalPairs++;}
        }
        // (Direct uploads into this response's slot: the batch's commands run after every response is recorded.)
        modeShareBuffer.SetData(modeShares,0,slot*modeShares.Length,modeShares.Length);modePairBuffer.SetData(modePairs,0,slot*modePairs.Length,modePairs.Length);
        UploadedBytes+=(modeShares.Length+modePairs.Length)*4L;
        if(ModalPairs==0)return false;
        modeAmplitudeBuffer.SetData(modeAmplitudes,0,slot*modeAmplitudes.Length,modeAmplitudes.Length);UploadedBytes+=modeAmplitudes.Length*4L;
        int size=modes.FftSize;
        Set("RoomModeCount",N);Set("RoomModeLength",modes.Length);Set("RoomModeSize",size);Set("RoomModeRatio",modes.Ratio);Set("RoomModeRadius",modes.Radius);
        commands.SetComputeFloatParam(shader,"RoomModeScale",(float)(1.0/((double)size*modes.OutputRate)));
        commands.SetComputeBufferParam(shader,modeSynthesis,"RoomModeTarget",modeFirst);
        commands.DispatchCompute(shader,modeSynthesis,(pairs*size+63)/64,1,1);
        var from=modeFirst;var to=modeSecond;
        void Transform(bool inverse)
        {
            Set("RoomModeInverse",inverse?1:0);
            for(int stage=1;stage<size;stage*=2)
            {
                commands.SetComputeBufferParam(shader,modeFft,"RoomModeSource",from);commands.SetComputeBufferParam(shader,modeFft,"RoomModeTarget",to);
                Set("RoomModeStage",stage);commands.DispatchCompute(shader,modeFft,(pairs*size/2+63)/64,1,1);
                (from,to)=(to,from);
            }
        }
        if(ModeStages==1){modeFinal=from;return true;}
        Transform(false);
        commands.SetComputeBufferParam(shader,modeCross,"RoomModeTarget",from);
        commands.DispatchCompute(shader,modeCross,(pairs*size+63)/64,1,1);
        Transform(true);
        modeFinal=from;
        return true;
    }
    // Profiling only (MeasureRoomModeCost): one update's modal work alone (the CPU probes and the kernels), submitted.
    internal void ProfileModes(AcousticTake.Scene scene,AcousticTake.State[] states){EnsurePath();poseCursor=1;if(SetModes(scene,states,true,true))AddModes();poseCursor=0;Submit();}
    // Checks only: the latest update's dh of one pair at the decimated rate (kernel samples, dh(t) / output rate).
    internal double[] ReadModes(int pair)
    {
        var data=new Vector2[modes.FftSize];modeFinal.GetData(data,0,pair*modes.FftSize,data.Length);
        var result=new double[data.Length];double scale=1.0/((double)modes.FftSize*modes.OutputRate);
        for(int j=0;j<data.Length;j++)result[j]=data[j].x*scale;
        return result;
    }
    // dh into the filtered partitions (after PathFilter, before PathPieces).
    void AddModes()
    {
        commands.SetComputeBufferParam(shader,modeAdd,"RoomModeSource",modeFinal);
        commands.DispatchCompute(shader,modeAdd,checked(pairs*lateCount*latePartition+63)/64,1,1);
    }
}
