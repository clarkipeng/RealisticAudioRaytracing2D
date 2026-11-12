using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// A late energy histogram [frame][pair][head-frame direction][band][bin] of
// (low, high) 32.32 sums, energy = value / LateRays, with optional plane emission
// rows (null for point sources only). RayTransport (hybrid) and MethodTransport
// (energy method) both provide one; the audio backends read either.
internal interface ILateField
{
    uint[] Late {get;}
    ComputeBuffer LateBuffer {get;}
    int LateUintsPerFrame {get;}
    // Cell offset, within a frame, of the room-energy block [pair][band][bin] (energy of all rays
    // at each bin start), or -1 without one.
    int LateRoomCells {get;}
    int LateBins {get;}
    int LateRays {get;}
    Emission.Batch Emissions {get;}
    ComputeBuffer EmissionRanges {get;}
    ComputeBuffer EmissionPoints {get;}
    ComputeBuffer EmissionClocks {get;}
}
// Shared 2D/3D GPU transport: phase-exact direct/early arrivals and the late energy histogram.
internal sealed class RayTransport : IDisposable, ILateField
{
    public const int PathStride = 48;
    // The same prefix ranges own transport paths, CPU groups and GPU FFT rows.
    // One plane's site capacity must not pad every other source/listener pair.
    internal sealed class Layout
    {
        internal readonly Vector2Int[] Ranges,DirectSites;
        internal readonly int[] Owners,RayCounts,GroupLimits;
        internal readonly int Capacity,MaximumSlots;
        internal Layout(AcousticTake.Scene scene,TakeRender.Settings settings)
        {
            Ranges=new Vector2Int[checked(scene.sources.Length*scene.listeners.Length)];
            RayCounts=new int[Ranges.Length];
            GroupLimits=new int[Ranges.Length];
            bool Constant(float[] values)
            {if(values!=null)for(int i=1;i<values.Length;i++)if(values[i]!=values[0])return false;return true;}
            bool sharedGeometry=true;
            foreach(var material in scene.materials)sharedGeometry&=Constant(material.ior);
            var direct=new List<Vector2Int>();
            // Energy method: each pair owns methodArrivalSlots discrete arrivals, one path per FFT bin.
            bool energy=settings.method==TakeRender.Method.Energy || settings.method==TakeRender.Method.Path;
            for(int s=0;s<scene.sources.Length;s++)
            {
                int sites=scene.planes==null?1:Array.Find(scene.planes,p=>p.source==s)?.samples??1;
                int rays=energy || settings.RaysPerPair==0?0:checked(settings.FrequencyBins*settings.SourceRaysPerBin(scene,s));
                int slots=energy?checked(settings.FrequencyBins*settings.ArrivalSlots):
                    checked(sites*settings.FrequencyBins*(1+settings.MirrorCandidates(scene)+settings.CompleteSlots(scene))+rays*settings.PathSlotsPerRay);
                int geometricBins=sharedGeometry?1:settings.FrequencyBins;
                int mirrors=settings.MirrorCandidates(scene)*sites;
                if(settings.reflectionSampling==TakeRender.ReflectionSampling.PrimitiveBaseline)mirrors=rays/settings.FrequencyBins;
                // Frequency-dependent loss changes pressure, not route geometry;
                // only a frequency-dependent speed splits a route into per-bin groups.
                int groups=energy?settings.ArrivalSlots:checked((sites+mirrors)*geometricBins+sites*settings.CompleteSlots(scene));
                MaximumSlots=Math.Max(MaximumSlots,slots);
                for(int l=0;l<scene.listeners.Length;l++)
                {
                    int pair=s*scene.listeners.Length+l;Ranges[pair]=new Vector2Int(Capacity,slots);Capacity=checked(Capacity+slots);
                    RayCounts[pair]=rays;
                    GroupLimits[pair]=groups;
                    for(int n=0;n<sites;n++)direct.Add(new Vector2Int(pair,n));
                }
            }
            Owners=new int[Capacity];for(int p=0;p<Ranges.Length;p++)Array.Fill(Owners,p,Ranges[p].x,Ranges[p].y);
            DirectSites=direct.ToArray();
        }
    }
    // One solved frame's late energy and emission rows, read by the audio backends.
    internal readonly struct LateFrame
    {
        internal readonly ILateField Transport;
        internal readonly int Frame;
        internal LateFrame(ILateField transport,int frame){Transport=transport;Frame=frame;}
        internal bool Valid => Transport!=null && Transport.LateUintsPerFrame>0;
        // Energy of one histogram cell (direct-sound units, (ReferenceDistance/distance)^2).
        internal double Energy(int pair,int direction,int band,int bin)
        {
            int cell=Frame*Transport.LateUintsPerFrame/2+(pair*TakeRender.Settings.LateDirections+direction)*TakeRender.Settings.LateBands*Transport.LateBins+band*Transport.LateBins+bin;
            return (Transport.Late[2*cell+1]+Transport.Late[2*cell]/4294967296.0)/Transport.LateRays;
        }
        internal bool HasRoom => Transport.LateRoomCells>=0;
        // Room energy (all rays) at the start of a bin, arbitrary units per pair and band.
        internal double Room(int pair,int band,int bin)
        {
            int cell=Frame*Transport.LateUintsPerFrame/2+Transport.LateRoomCells+(pair*TakeRender.Settings.LateBands+band)*Transport.LateBins+bin;
            return Transport.Late[2*cell+1]+Transport.Late[2*cell]/4294967296.0;
        }
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Path
    {
        public Vector4 direction; // xyz arrival direction; w = d(propagation seconds)/d(receiver second)
        public float delaySeconds, pressureGain;
        public uint frequencyBin, identity;
        public float pressureImag;
        public uint emissionPhase, birthLow, birthHigh; // High phase bit selects finite burst; otherwise legacy 16-bit phase.
        public long Birth => (long)(((ulong)birthHigh<<32)|birthLow);
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Material
    {
        public Vector4 surface; // absorption, scattering, transmission, IOR
        public Vector4 medium;  // damping; reserved
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Edge
    {
        public Vector4 a, b, face, normal; // face.w: exterior wedge angle; normal points into air.
        public uint owner, firstFace, secondFace, pad;
    }
    readonly ComputeShader shader;
    readonly CommandBuffer commands;
    readonly int directKernel, baselineKernel, pairs, frequencyBins;
    readonly int mirrorKernel;
    readonly int prepareComplete,compactComplete,integrateComplete,completeSamples,completeCapacity,pathSlots,completeSlots;
    readonly ComputeBuffer completeCandidates,completeSelected,completeCount,completeDraws,completeSteps;
    readonly bool visibleMirrors;
    readonly int[] mirrorCounts,mirrorSites;
    // Opt-in per-family gains; null renders the unbiased estimator unchanged.
    readonly FamilyGainTable familyGains;
    readonly Vector4 familyGainK0,familyGainK1;
    readonly int gainMirrors,gainComplete;
    readonly int edgeCount,primitiveCount,maxBounces;
    readonly float diffractionFactor;
    internal FamilyGainTable FamilyGains => familyGains;
    readonly Layout layout;
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal struct Volume
    {
        public Vector4 minimum, maximum; // w: winding correction, enclosed volume/area (relative units)
        public uint start, count, owner, material;
    }
    readonly ComputeBuffer materialBuffer, edgeBuffer, volumeBuffer, pathBuffer, enabledBuffer,
        edgeRangeBuffer, edgeIndexBuffer, emissionRanges, emissionPoints, emissionClocks, pairRanges, directSites;
    readonly Emission.Batch emissions;
    readonly int lateKernel,clearLate,lateRays;
    readonly ComputeBuffer lateHistogram,lateMaterials,lateSites,lateSiteRanges;
    // Late energy histogram: per frame, pair, head-frame direction, band and bin, a
    // (low, high) 32.32 fixed-point sum of ray shares; energy = value / LateRays.
    public readonly uint[] Late;
    uint[] ILateField.Late => Late;
    public int LateBins {get;}
    internal int LateCells {get;}
    public int LateUintsPerFrame {get;}
    public int LateRoomCells => -1;
    public int LateRays => lateRays;
    public ComputeBuffer LateBuffer => lateHistogram;
    public Emission.Batch Emissions => emissions;
    public ComputeBuffer EmissionRanges => emissionRanges;
    public ComputeBuffer EmissionPoints => emissionPoints;
    public ComputeBuffer EmissionClocks => emissionClocks;
    readonly bool[] everActive;
    readonly uint[] enabledPairs;
    readonly Action<AsyncGPUReadbackRequest> completion;
    readonly Action<AsyncGPUReadbackRequest> lateCompletion;
    AsyncGPUReadbackRequest request,lateRequest;
    bool pending, failed;
    volatile bool completed,lateCompleted;
    public readonly Path[] Paths;
    public int PathsPerFrame { get; }
    public int SlotsPerPair { get; }
    internal ComputeBuffer Buffer => pathBuffer;
    public long CpuBytes => Paths.Length * (long)PathStride + Late.Length * 4L + enabledPairs.Length * 4L + everActive.Length + emissions.NumericBytes + (layout.Owners.Length+layout.RayCounts.Length+layout.GroupLimits.Length+mirrorCounts.Length+mirrorSites.Length)*4L + (layout.Ranges.Length+layout.DirectSites.Length)*8L;
    public long GpuBytes { get; }
    public RayTransport(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
                        TakeRender.Settings settings, ComputeShader shader, ComputeBuffer primitives,
                        ComputeBuffer states, ComputeBuffer sources, ComputeBuffer listeners,
                        ComputeBuffer bvhNodes, ComputeBuffer bvhIndices, int rate, int maximumFrames,
                        bool readback = true)
    {
        if (Marshal.SizeOf<Path>() != PathStride || Marshal.SizeOf<Material>() != 32 ||
            Marshal.SizeOf<Edge>() != 80 || Marshal.SizeOf<Volume>() != 48)
            throw new InvalidOperationException("Spectral GPU ABI mismatch.");
        this.shader = shader;
        int declaredPrimitives=0;foreach(var obj in scene.objects)declaredPrimitives=checked(declaredPrimitives+obj.primitiveCount);
        if(declaredPrimitives!=geometry.Length)throw new ArgumentException("Scene primitive ranges must cover the supplied transport geometry.");
        frequencyBins = settings.fftSize / 2 + 1;
        completeSamples=settings.CompleteSamples;completeSlots=settings.CompleteSlots(scene);pathSlots=settings.PathSlotsPerRay;
        pairs = checked(scene.sources.Length * scene.listeners.Length);
        visibleMirrors=settings.reflectionSampling==TakeRender.ReflectionSampling.FirstImages;
        TakeRender.ValidateFamilyGains(settings);
        familyGains=settings.FamilyGains;
        if(familyGains!=null)
        {
            familyGainK0=new Vector4(familyGains.k[0],familyGains.k[1],familyGains.k[2],familyGains.k[3]);
            familyGainK1=new Vector4(familyGains.k[4],familyGains.k[5],0,0);
        }
        primitiveCount=geometry.Length;maxBounces=settings.maxBounces;diffractionFactor=settings.diffractionFactor;
        mirrorCounts=new int[pairs];mirrorSites=new int[pairs];
        everActive = new bool[scene.sources.Length];
        enabledPairs = new uint[checked(pairs * maximumFrames)];
        int bins = settings.fftSize / 2 + 1;
        foreach(var material in scene.materials)material.ValidateAdmittance();
        layout=new Layout(scene,settings);SlotsPerPair=layout.MaximumSlots;PathsPerFrame=layout.Capacity;
        Paths = new Path[readback ? checked(PathsPerFrame * maximumFrames) : 0];
        completion = OnComplete;lateCompletion=OnLateComplete;
        bool planar = scene.dimension == AcousticTake.Dimension.Planar2D;
        try
        {
            commands = new CommandBuffer {name="Acoustics GPU transport"};
            pairRanges=new ComputeBuffer(layout.Ranges.Length,8);pairRanges.SetData(layout.Ranges);
            directSites=new ComputeBuffer(layout.DirectSites.Length,8);directSites.SetData(layout.DirectSites);
            emissions=new Emission.Batch(scene,settings,rate,maximumFrames,geometry);
            emissionRanges=new ComputeBuffer(emissions.Ranges.Length,16);emissionRanges.SetData(emissions.Ranges);
            emissionPoints=new ComputeBuffer(emissions.Points.Length,16);emissionPoints.SetData(emissions.Points);
            emissionClocks=new ComputeBuffer(emissions.Clocks.Length,16);emissionClocks.SetData(emissions.Clocks);
            directKernel = shader.FindKernel((planar ? "EvaluateDirect2D" : "EvaluateDirect3D") +
                                             (settings.useBvh ? "Bvh" : ""));
            baselineKernel =
                shader.FindKernel((planar ? "BaselineMirrors2D" : "BaselineMirrors3D") + (settings.useBvh ? "Bvh" : ""));
            lateKernel=shader.FindKernel((planar?"TraceLate2D":"TraceLate3D")+(settings.useBvh?"Bvh":""));clearLate=shader.FindKernel("ClearLate");
            mirrorKernel=shader.FindKernel((planar?"PrepareMirrors2D":"PrepareMirrors3D")+(settings.useBvh?"Bvh":""));
            prepareComplete=shader.FindKernel((planar?"PrepareComplete2D":"PrepareComplete3D")+(settings.useBvh?"Bvh":""));
            integrateComplete=shader.FindKernel((planar?"IntegrateComplete2D":"IntegrateComplete3D")+(settings.useBvh?"Bvh":""));
            compactComplete=shader.FindKernel("CompactComplete");
            gainMirrors=shader.FindKernel("GainMirrors");gainComplete=shader.FindKernel(planar?"GainComplete2D":"GainComplete3D");
            if(familyGains!=null && (!shader.IsSupported(gainMirrors) || !shader.IsSupported(gainComplete)))
                throw new NotSupportedException("Family gain kernels are unavailable on this GPU.");
            if(completeSamples>0 && (!shader.IsSupported(prepareComplete) || !shader.IsSupported(compactComplete) || !shader.IsSupported(integrateComplete)))
                throw new NotSupportedException("Complete-route kernels are unavailable on this GPU.");
            if (!shader.IsSupported(directKernel) || !shader.IsSupported(baselineKernel) || !shader.IsSupported(lateKernel) || !shader.IsSupported(clearLate))
                throw new NotSupportedException("Raytraced kernels are unavailable on this GPU.");
            if(visibleMirrors && !shader.IsSupported(mirrorKernel))
                throw new NotSupportedException("Complete-route kernels are unavailable on this GPU.");
            var materials = new Material[Math.Max(1, scene.materials.Length * bins)];
            for (int m = 0; m < scene.materials.Length; m++)
                for (int b = 0; b < bins; b++)
                {
                    var material = scene.materials[m];
                    float hz = (float)b * rate / settings.fftSize;
                    materials[m * bins + b] =
                        new Material { surface = new Vector4(Sample(material, material.absorption, hz, 0),
                                                             Sample(material, material.scattering, hz, 0),
                                                             Sample(material, material.transmission, hz, 0),
                                                             Sample(material, material.ior, hz, 1)),
                                       medium =
                                           new Vector4(Sample(material, material.damping, hz, 0),
                                               Sample(material, material.admittanceReal, hz, 0),
                                               Sample(material, material.admittanceImag, hz, 0, true), material.HasAdmittance?1:0) };
                }
            materialBuffer = new ComputeBuffer(materials.Length, 32);
            materialBuffer.SetData(materials);
            var volumes = Volumes(scene, geometry, materials, bins, planar);
            var edges = Edges(geometry, planar, materials, bins, volumes);edgeCount=edges.Length;
            long candidates=edges.Length+(settings.maxBounces>1?(long)edges.Length*edges.Length+2L*geometry.Length*edges.Length+(long)geometry.Length*geometry.Length:0);
            completeCapacity=(int)Math.Max(1,Math.Min(completeSamples,candidates));
            completeCandidates=new ComputeBuffer(completeCapacity,16);completeSelected=new ComputeBuffer(completeCapacity,4);
            completeSteps=new ComputeBuffer(completeSamples>0&&planar?checked(2*completeCapacity):1,48);
            completeCount=new ComputeBuffer(1,4);completeDraws=new ComputeBuffer(Math.Max(1,completeSamples),16);
            if(completeSamples>0)completeDraws.SetData(RouteDraws(completeSamples,settings.seed));
            edgeBuffer = new ComputeBuffer(Math.Max(1, edges.Length), 80);
            if (edges.Length > 0)
                edgeBuffer.SetData(edges);
            volumeBuffer = new ComputeBuffer(Math.Max(1, volumes.Length), 48);
            if (volumes.Length > 0)
                volumeBuffer.SetData(volumes);
            pathBuffer = new ComputeBuffer(checked(PathsPerFrame * maximumFrames), PathStride);
            for(int p=0;p<pairs;p++)
            {
                int source=p/scene.listeners.Length;
                int sites=scene.planes==null?1:Array.Find(scene.planes,x=>x.source==source)?.samples??1;
                int count=settings.MirrorCandidates(scene);
                mirrorCounts[p]=count;mirrorSites[p]=sites;
            }
            enabledBuffer = new ComputeBuffer(enabledPairs.Length, 4);
            var adjacency = new List<int>[geometry.Length];
            for (int i = 0; i < adjacency.Length; i++)
                adjacency[i] = new List<int>();
            for (int i = 0; i < edges.Length; i++)
            {
                adjacency[edges[i].firstFace].Add(i);
                if (edges[i].secondFace != edges[i].firstFace)
                    adjacency[edges[i].secondFace].Add(i);
            }
            var edgeIndices = new List<int>();
            var edgeRanges = new Vector2Int[Math.Max(1, geometry.Length)];
            for (int i = 0; i < adjacency.Length; i++)
            {
                edgeRanges[i] = new Vector2Int(edgeIndices.Count, adjacency[i].Count);
                edgeIndices.AddRange(adjacency[i]);
            }
            edgeRangeBuffer = new ComputeBuffer(edgeRanges.Length, 8);
            edgeRangeBuffer.SetData(edgeRanges);
            edgeIndexBuffer = new ComputeBuffer(Math.Max(1, edgeIndices.Count), 4);
            if (edgeIndices.Count > 0)
                edgeIndexBuffer.SetData(edgeIndices);
            // Late field: band-centre materials, stable emission sites and the
            // fixed-point energy histogram (see LateEnergy.hlsl).
            var bandMaterials=new Material[Math.Max(1,scene.materials.Length*TakeRender.Settings.LateBands)];
            for(int m=0;m<scene.materials.Length;m++)for(int b=0;b<TakeRender.Settings.LateBands;b++)
            {
                var material=scene.materials[m];float hz=125f*(1<<b);
                bandMaterials[m*TakeRender.Settings.LateBands+b]=new Material{surface=new Vector4(Sample(material,material.absorption,hz,0),Sample(material,material.scattering,hz,0),Sample(material,material.transmission,hz,0),Sample(material,material.ior,hz,1)),
                    medium=new Vector4(Sample(material,material.damping,hz,0),Sample(material,material.admittanceReal,hz,0),Sample(material,material.admittanceImag,hz,0,true),material.HasAdmittance?1:0)};
            }
            lateMaterials=new ComputeBuffer(bandMaterials.Length,32);lateMaterials.SetData(bandMaterials);
            lateRays=settings.lateRays;LateBins=settings.LateBins(rate);
            LateCells=TakeRender.Settings.LateDirections*TakeRender.Settings.LateBands*LateBins;
            LateUintsPerFrame=checked(2*pairs*LateCells);
            lateHistogram=new ComputeBuffer(Math.Max(1,checked(LateUintsPerFrame*maximumFrames)),4);
            Late=new uint[readback?checked(LateUintsPerFrame*maximumFrames):0];
            lateSites=new ComputeBuffer(Math.Max(1,emissions.LateSites.Length),16);lateSiteRanges=new ComputeBuffer(emissions.LateRanges.Length,16);
            UploadLateSites();
            foreach (int kernel in new[] { directKernel, baselineKernel, mirrorKernel,prepareComplete,compactComplete,integrateComplete,gainMirrors,gainComplete,lateKernel,clearLate })
            {
                shader.SetBuffer(kernel,"CompleteCandidates",completeCandidates);shader.SetBuffer(kernel,"CompleteSelected",completeSelected);
                shader.SetBuffer(kernel,"CompleteCount",completeCount);shader.SetBuffer(kernel,"CompleteDraws",completeDraws);shader.SetBuffer(kernel,"CompleteSteps",completeSteps);
                shader.SetBuffer(kernel, "Primitives", primitives);
                shader.SetBuffer(kernel, "States", states);
                shader.SetBuffer(kernel, "Sources", sources);
                shader.SetBuffer(kernel, "RayPairRanges", pairRanges);
                shader.SetBuffer(kernel, "RayDirectSites", directSites);
                shader.SetBuffer(kernel, "EmissionRanges", emissionRanges);
                shader.SetBuffer(kernel, "EmissionPoints", emissionPoints);
                shader.SetBuffer(kernel, "EmissionClocks", emissionClocks);
                shader.SetBuffer(kernel, "Listeners", listeners);
                shader.SetBuffer(kernel, "RayMaterials", materialBuffer);
                shader.SetBuffer(kernel, "Edges", edgeBuffer);
                shader.SetBuffer(kernel, "EdgeRanges", edgeRangeBuffer);
                shader.SetBuffer(kernel, "EdgeIndices", edgeIndexBuffer);
                shader.SetBuffer(kernel, "Volumes", volumeBuffer);
                shader.SetBuffer(kernel, "RayPaths", pathBuffer);
                shader.SetBuffer(kernel, "RayPairEnabled", enabledBuffer);
                shader.SetBuffer(kernel,"LateMaterials",lateMaterials);shader.SetBuffer(kernel,"LateSites",lateSites);
                shader.SetBuffer(kernel,"LateSiteRanges",lateSiteRanges);shader.SetBuffer(kernel,"LateHistogram",lateHistogram);
                if (bvhNodes != null)
                {
                    shader.SetBuffer(kernel, "BvhNodes", bvhNodes);
                    shader.SetBuffer(kernel, "BvhIndices", bvhIndices);
                }
            }
            shader.SetInt("RaysPerBin", settings.RaysPerBin);
            shader.SetInt("UseVisibleMirrors",visibleMirrors?1:0);
            shader.SetInt("CompleteSamples",completeSamples);shader.SetInt("CompleteCapacity",completeCapacity);
            shader.SetInt("PathSlotsPerRay",settings.PathSlotsPerRay);
            shader.SetInt("RaySeed", settings.seed);
            shader.SetInt("FrequencyBins", bins);
            shader.SetInt("RaysPerPair", settings.RaysPerPair);
            shader.SetInt("MaxBounces", settings.maxBounces);
            shader.SetInt("EmissionFrameStride", emissions.FrameStride);
            shader.SetInt("RayPathsPerFrame", PathsPerFrame);
            shader.SetInt("RayDirectSiteCount", layout.DirectSites.Length);
            shader.SetInt("EdgeCount", edges.Length);
            shader.SetInt("VolumeCount", volumes.Length);
            shader.SetFloat("FrequencyStep", (float)rate / settings.fftSize);
            shader.SetFloat("DiffractionFactor", settings.diffractionFactor);
            shader.SetFloat("MaximumPathDelay", settings.maximumDelaySeconds);
            shader.SetInt("UseAirAbsorption", settings.airAbsorption ? 1 : 0);
            shader.SetInt("LateRays",lateRays);shader.SetInt("LateBins",LateBins);shader.SetInt("LatePairCells",LateCells);
            shader.SetInt("LateEarlyOrder",settings.LateEarlyOrder);shader.SetFloat("LateReceiverRadius",settings.lateReceiverRadius);
            shader.SetFloat("LateBinSeconds",(float)((double)settings.LatePartition/rate));
            GpuBytes = (long)completeSteps.count*48+(long)completeCapacity*20+4+(long)Math.Max(1,completeSamples)*16 + (long)edgeRangeBuffer.count * 8 + (long)edgeIndexBuffer.count * 4 +
                       (long)materials.Length * 32 + (long)Math.Max(1, edges.Length) * 80 +
                       Math.Max(1, volumes.Length) * 48L + (long)PathsPerFrame * maximumFrames * PathStride +
                       enabledPairs.Length * 4L + (long)(emissions.Ranges.Length+2*emissions.Points.Length)*16+(long)(layout.Ranges.Length+layout.DirectSites.Length)*8+
                       (long)lateHistogram.count*4+(long)(lateMaterials.count*2+lateSites.count+lateSiteRanges.count)*16;
        }
        catch
        {
            Dispose();
            throw;
        }
    }
    void UploadLateSites()
    {
        if(emissions.LateSites.Length>0)lateSites.SetData(emissions.LateSites);
        lateSiteRanges.SetData(emissions.LateRanges);
    }
    internal void PrepareEmissions(TakeRender.Signal[] signals,long first,int frames,Action<long,AcousticTake.State[]> evaluate)
    {
        if(emissions.Prepare(signals,first,frames,evaluate))UploadLateSites();
        emissionRanges.SetData(emissions.Ranges,0,0,frames*emissions.FrameStride);
        emissionPoints.SetData(emissions.Points,0,0,frames*emissions.PointsPerFrame);emissionClocks.SetData(emissions.Clocks,0,0,frames*emissions.PointsPerFrame);
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct CompleteRoute { public uint first, second, family; public float weight; }
    // Diagnostic synchronous export for offline datasets. Returns the complete-route
    // family (0 D, 1 DD, 2 RD, 3 DR, 4 RR) of each integration sample written by the
    // latest dispatch, or -1 where no route was selected. It reads the route tables
    // left by that dispatch and repeats IntegrateCompleteRoutes' sample-to-route
    // mapping, so it is defined only for one pair, one frame and one emission site.
    internal int[] CompleteSampleFamilies()
    {
        if (pending || pairs != 1 || mirrorSites[0] != 1 || completeSlots != completeSamples)
            throw new InvalidOperationException("Complete-route families need one idle 3D point-source pair.");
        var families = new int[completeSamples];
        Array.Fill(families, -1);
        if (completeSamples == 0)
            return families;
        var count = new uint[1];
        completeCount.GetData(count);
        if (count[0] == 0)
            return families;
        if (count[0] > completeCapacity)
            throw new InvalidOperationException("Complete-route count exceeds its capacity.");
        var selected = new uint[count[0]];
        completeSelected.GetData(selected, 0, 0, selected.Length);
        var routes = new CompleteRoute[completeCapacity];
        completeCandidates.GetData(routes);
        uint total = (uint)completeSamples, per = total / count[0], extra = total % count[0], split = (per + 1) * extra;
        for (uint index = 0; index < total; index++)
        {
            uint rank = index < split ? index / (per + 1) : extra + (index - split) / per;
            families[index] = (int)routes[selected[rank]].family;
        }
        return families;
    }
    // CPU mirror of the shader's rho for ApplyFamilyGain (family 0 first-order images,
    // 1-5 complete D, DD, RD, DR, RR), for checks and diagnostics.
    internal float RoutesPerSample(int family)
    {
        if(family==0)return (float)primitiveCount/Math.Max(1,mirrorCounts[0]);
        float e=edgeCount,p=primitiveCount;bool second=maxBounces>1;
        var sizes=diffractionFactor<=0?new float[4]:new[]{e,second?e*e:0,second?p*e:0,second?p*e:0};
        float pairs=second?p*p:0,total=sizes[0]+sizes[1]+sizes[2]+sizes[3]+pairs;
        if(total<=completeCapacity)return total/completeSamples;
        int families=(pairs>0?1:0);foreach(float size in sizes)families+=size>0?1:0;
        return (family<5?sizes[family-1]:pairs)*families/completeCapacity;
    }
    internal int EdgeCount => edgeCount;
    internal void BindDebug(ComputeBuffer segments)
    {
        shader.SetBuffer(directKernel, "DebugSegments", segments);
        shader.SetBuffer(baselineKernel, "DebugSegments", segments);
        shader.SetBuffer(mirrorKernel,"DebugSegments",segments);
        shader.SetBuffer(integrateComplete,"DebugSegments",segments);
    }
    public void SetActivity(AcousticTake.Scene scene, AcousticTake.State[] states, int frames)
    {
        // Never-used slots and permanently muted pairs cannot contribute. Once activated,
        // keep tracing a source after it leaves: earlier emission can still be arriving.
        for (int f = 0; f < frames; f++)
            for (int s = 0; s < everActive.Length; s++)
            {
                everActive[s] |=
                    TakeRender.Enabled(states[f * scene.objects.Length + scene.sources[s].objectIndex],
                                       AcousticTake.StateFlags.Source);
                for (int l = 0; l < scene.listeners.Length; l++)
                    enabledPairs[f * pairs + s * scene.listeners.Length + l] =
                        everActive[s] && scene.sources[s].mutedListenerPlusOne != l + 1 ? 1u : 0u;
            }
        enabledBuffer.SetData(enabledPairs, 0, 0, frames * pairs);
    }
    // Closed, consistently wound meshes/loops have thickness. Open sheets only
    // apply their surface coefficients. This is derived once, not saved in takes.
    internal static Volume[] Volumes(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry, Material[] materials,
                            int bins, bool planar)
    {
        var result = new List<Volume>();
        for (int owner = 0; owner < scene.objects.Length; owner++)
        {
            var obj = scene.objects[owner];
            if (obj.primitiveCount == 0)
                continue;
            int start = obj.primitiveStart, end = start + obj.primitiveCount;
            var min = geometry[start].a;
            var max = min;
            for (int i = start; i < end; i++)
            {
                var p = geometry[i];
                min = Vector3.Min(min, Vector3.Min(p.a, p.b));
                max = Vector3.Max(max, Vector3.Max(p.a, p.b));
                if (!planar)
                {
                    min = Vector3.Min(min, p.c);
                    max = Vector3.Max(max, p.c);
                }
            }
            // Weld floating-point seams from curved colliders for topology only.
            float tolerance = Mathf.Max(1e-7f, (max - min).magnitude * 1e-6f);
            var vertices = new Dictionary<Vector3Int, int>();
            int Vertex(Vector3 p)
            {
                p = (p - min) / tolerance;
                var key = new Vector3Int(Mathf.RoundToInt(p.x), Mathf.RoundToInt(p.y),
                                         planar ? 0 : Mathf.RoundToInt(p.z));
                if (!vertices.TryGetValue(key, out int id))
                {
                    id = vertices.Count;
                    vertices.Add(key, id);
                }
                return id;
            }
            var edges = new Dictionary<(int, int), (int count, int winding)>();
            void Edge(int a, int b)
            {
                var key = a < b ? (a, b) : (b, a);
                edges.TryGetValue(key, out var value);
                edges[key] = (value.count + 1, value.winding + (a < b ? 1 : -1));
            }
            double signed = 0;
            var degree = new Dictionary<int, (int incoming, int outgoing)>();
            for (int i = start; i < end; i++)
            {
                var p = geometry[i];
                int a = Vertex(p.a), b = Vertex(p.b);
                if (planar)
                {
                    degree.TryGetValue(a, out var va);
                    degree[a] = (va.incoming, va.outgoing + 1);
                    degree.TryGetValue(b, out var vb);
                    degree[b] = (vb.incoming + 1, vb.outgoing);
                    signed += (double)p.a.x * p.b.y - (double)p.a.y * p.b.x;
                }
                else
                {
                    int c = Vertex(p.c);
                    Edge(a, b);
                    Edge(b, c);
                    Edge(c, a);
                    signed += Vector3.Dot(p.a - min, Vector3.Cross(p.b - min, p.c - min));
                }
            }
            bool closed = signed != 0;
            foreach (var edge in edges.Values)
                closed &= edge.count == 2 && edge.winding == 0;
            foreach (var vertex in degree.Values)
                closed &= vertex.incoming == 1 && vertex.outgoing == 1;
            if (!closed)
                continue;
            int material = geometry[start].materialIndex;
            for (int i = start + 1; i < end; i++)
                for (int b = 0; b < bins; b++)
                {
                    var a = materials[material * bins + b];
                    var other = materials[geometry[i].materialIndex * bins + b];
                    if (a.surface.w != other.surface.w || a.medium.x != other.medium.x)
                        throw new ArgumentException(obj.name +
                                                    (": a closed acoustic object must share interior " +
                                                     "IOR/damping across its surface materials."));
                }
            result.Add(new Volume {
                minimum = new Vector4(min.x, min.y, min.z, (signed > 0 ? 1 : -1) * (planar ? -1 : 1)),
                maximum = new Vector4(max.x, max.y, max.z, (float)Math.Abs(signed)), start = (uint)start,
                count = (uint)obj.primitiveCount, owner = (uint)owner, material = (uint)material
            });
        }
        return result.ToArray();
    }
    internal static float Sample(AcousticTake.MaterialDefinition material, float[] values, float frequency,
                        float fallback, bool signed=false)
    {
        if (values == null || values.Length == 0)
            return fallback;
        var knots = material.frequencies;
        if (knots == null || knots.Length != values.Length)
            throw new ArgumentException("Material coefficient grids do not match.");
        int upper = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]) || (!signed && values[i] < 0) || !float.IsFinite(knots[i]) || knots[i] < 0 ||
                (i > 0 && knots[i] <= knots[i - 1]))
                throw new ArgumentException("Invalid material coefficient curve.");
        }
        while (upper < knots.Length - 1 && frequency > knots[upper])
            upper++;
        int lower = Math.Max(0, upper - 1);
        return Mathf.Lerp(values[lower], values[upper],
                          lower == upper ? 0 : Mathf.InverseLerp(knots[lower], knots[upper], frequency));
    }
    internal static Edge[] Edges(AcousticTake.Primitive[] geometry, bool planar, Material[] materials, int bins,
                        Volume[] volumes)
    {
        var winding = new Dictionary<uint, float>();
        foreach (var volume in volumes)
            winding[volume.owner] = volume.minimum.w;
        var candidates = new Dictionary<(int, Vector3, Vector3),
                                        (Vector3 normal, int first, int second, bool keep, int count)>();
        int Compare(Vector3 a, Vector3 b)
        {
            int r = a.x.CompareTo(b.x);
            if (r == 0)
                r = a.y.CompareTo(b.y);
            return r == 0 ? a.z.CompareTo(b.z) : r;
        }
        void Add(int face, Vector3 a, Vector3 b, Vector3 normal)
        {
            if (Compare(a, b) > 0)
            {
                var swap = a;
                a = b;
                b = swap;
            }
            var key = (geometry[face].objectIndex, a, b);
            if (candidates.TryGetValue(key, out var old))
                candidates[key] =
                    (old.normal, old.first, face,
                     old.count == 1 && Mathf.Abs(Vector3.Dot(old.normal, normal)) < .99999f, old.count + 1);
            else
                candidates.Add(key, (normal, face, face, true, 1));
        }
        for (int i = 0; i < geometry.Length; i++)
        {
            var p = geometry[i];
            if (planar)
            {
                var n = new Vector3(-(p.b - p.a).y, (p.b - p.a).x, 0).normalized;
                Add(i, p.a, p.a, n);
                Add(i, p.b, p.b, n);
            }
            else
            {
                var n = Vector3.Cross(p.b - p.a, p.c - p.a).normalized;
                Add(i, p.a, p.b, n);
                Add(i, p.b, p.c, n);
                Add(i, p.c, p.a, n);
            }
        }
        bool Opaque(int face)
        {
            int start = geometry[face].materialIndex * bins;
            for (int bin = 0; bin < bins; bin++)
            {
                var m = materials[start + bin].surface;
                if ((1 - m.x) * m.z < 1)
                    return true;
            }
            return false;
        }
        Vector3 Side(int face, Vector3 a, Vector3 b)
        {
            var p = geometry[face];
            if (planar)
                return ((p.a - a).sqrMagnitude > 1e-12f ? p.a - a : p.b - a).normalized;
            var axis = (b - a).normalized;
            var side = p.a + p.b + p.c - 3 * a;
            return (side - axis * Vector3.Dot(side, axis)).normalized;
        }
        var result = new List<Edge>();
        foreach (var item in candidates)
        {
            var value = item.Value;
            if (!value.keep || (!Opaque(value.first) && !Opaque(value.second)))
                continue;
            var a = item.Key.Item2;
            var b = item.Key.Item3;
            var side = Side(value.first, a, b);
            var normal = value.normal;
            if (winding.TryGetValue((uint)item.Key.Item1, out float sign))
                normal *= sign;
            float angle = 2 * Mathf.PI;
            if (value.first != value.second)
            {
                var other = Side(value.second, a, b);
                angle = Mathf.Atan2(Vector3.Dot(other, normal), Vector3.Dot(other, side));
                if (angle <= 0)
                    angle += 2 * Mathf.PI;
            }
            result.Add(new Edge { owner = (uint)item.Key.Item1, a = a, b = b,
                                  face = new Vector4(side.x, side.y, side.z, angle), normal = normal,
                                  firstFace = (uint)value.first, secondFace = (uint)value.second });
        }
        result.Sort((a, b) =>
                    {
                        int r = a.owner.CompareTo(b.owner);
                        if (r == 0)
                            r = Compare(a.a, b.a);
                        return r == 0 ? Compare(a.b, b.b) : r;
                    });
        return result.ToArray();
    }
    static Vector4[] RouteDraws(int count,int seed)
    {
        var draws=new Vector4[count];int[] bases={2,3,5,7};var shifts=new double[4];
        uint state=unchecked((uint)seed)^0x53a91u;
        for(int d=0;d<4;d++){state^=state>>16;state*=0x7feb352du;state^=state>>15;state*=0x846ca68bu;state^=state>>16;shifts[d]=(state+.5)/4294967296.0;state+=0x9e3779b9u;}
        for(int i=0;i<count;i++)for(int d=0;d<4;d++)
        {
            int n=i+1;double denominator=1,value=0;
            while(n>0){denominator*=bases[d];value+=(n%bases[d])/denominator;n/=bases[d];}
            value+=shifts[d];value-=Math.Floor(value);draws[i][d]=Mathf.Clamp((float)value,1e-7f,1-1e-7f);
        }
        return draws;
    }
    public void Dispatch(int frames, bool asynchronous)
    {
        if (pending)
            throw new InvalidOperationException("Ray responses are still in flight.");
        // Bound per dispatch: transports sharing this shader may differ.
        if(familyGains!=null)
        {
            commands.SetComputeVectorParam(shader,"FamilyGainK0",familyGainK0);commands.SetComputeVectorParam(shader,"FamilyGainK1",familyGainK1);
            commands.SetComputeFloatParam(shader,"FamilyGainPower",familyGains.power);
        }
        commands.BeginSample("Acoustics.GPU.Direct");
        commands.DispatchCompute(shader,directKernel,(checked(frames*layout.DirectSites.Length*frequencyBins)+63)/64,1,1);
        commands.EndSample("Acoustics.GPU.Direct");
        if(maxBounces>0)
        {
            commands.BeginSample("Acoustics.GPU.Early");
            for(int pair=0;pair<pairs;pair++)
            {
                commands.SetComputeIntParam(shader,"RayTracePair",pair);
                commands.SetComputeIntParam(shader,"RaysPerPair",layout.RayCounts[pair]);
                commands.SetComputeIntParam(shader,"RaysPerBin",Math.Max(1,layout.RayCounts[pair]/frequencyBins));
                commands.SetComputeIntParam(shader,"MirrorCandidateCount",mirrorCounts[pair]);
                commands.SetComputeIntParam(shader,"MirrorSiteCapacity",mirrorSites[pair]);
                if(mirrorCounts[pair]>0)
                {
                    int rows=checked(frames*mirrorSites[pair]*frequencyBins);
                    commands.DispatchCompute(shader,mirrorKernel,(checked(rows*mirrorCounts[pair])+63)/64,1,1);
                    if(familyGains!=null)commands.DispatchCompute(shader,gainMirrors,(checked(rows*mirrorCounts[pair])+63)/64,1,1);
                }
                if(layout.RayCounts[pair]>0)commands.DispatchCompute(shader,baselineKernel,(checked(frames*layout.RayCounts[pair])+63)/64,1,1);
                if(completeSamples>0)for(int frame=0;frame<frames;frame++)for(int site=0;site<mirrorSites[pair];site++)
                {
                    int first=checked(frame*PathsPerFrame+layout.Ranges[pair].x+mirrorSites[pair]*frequencyBins*(1+mirrorCounts[pair])+layout.RayCounts[pair]*pathSlots+site*completeSlots*frequencyBins);
                    commands.SetComputeIntParam(shader,"CompleteOffset",first);commands.SetComputeIntParam(shader,"CompleteFrame",frame);
                    commands.SetComputeIntParam(shader,"CompleteSite",site);
                    commands.DispatchCompute(shader,prepareComplete,(completeCapacity+63)/64,1,1);
                    commands.DispatchCompute(shader,compactComplete,1,1,1);
                    commands.DispatchCompute(shader,integrateComplete,(completeSamples+63)/64,1,1);
                    // The gain pass reads this site's route tables before the next site reuses them.
                    if(familyGains!=null)commands.DispatchCompute(shader,gainComplete,(checked(completeSlots*frequencyBins)+63)/64,1,1);
                }
            }
            commands.EndSample("Acoustics.GPU.Early");
            commands.BeginSample("Acoustics.GPU.Late");
            commands.DispatchCompute(shader,clearLate,(checked(frames*LateUintsPerFrame)+63)/64,1,1);
            for(int pair=0;pair<pairs;pair++)
            {
                commands.SetComputeIntParam(shader,"RayTracePair",pair);
                commands.DispatchCompute(shader,lateKernel,(checked(frames*lateRays*TakeRender.Settings.LateBands)+63)/64,1,1);
            }
            commands.EndSample("Acoustics.GPU.Late");
        }
        Graphics.ExecuteCommandBuffer(commands);commands.Clear();
        if (Paths.Length == 0)
            return;
        if (asynchronous)
        {
            pending = true;
            completed = lateCompleted = false;
            failed = false;
            request = AsyncGPUReadback.Request(pathBuffer, checked(frames * PathsPerFrame * PathStride), 0, completion);
            lateRequest=AsyncGPUReadback.Request(lateHistogram,checked(Math.Max(1,frames*LateUintsPerFrame)*4),0,lateCompletion);
        }
        else
        {
            pathBuffer.GetData(Paths, 0, 0, frames * PathsPerFrame);
            if(LateUintsPerFrame>0)lateHistogram.GetData(Late,0,0,frames*LateUintsPerFrame);
        }
    }
    void OnComplete(AsyncGPUReadbackRequest value)
    {
        failed = value.hasError;
        if (!failed)
        {
            var data = value.GetData<Path>();
            Unity.Collections.NativeArray<Path>.Copy(data, Paths, data.Length);
        }
        completed = true;
    }
    void OnLateComplete(AsyncGPUReadbackRequest value)
    {
        failed |= value.hasError;
        if (!value.hasError && LateUintsPerFrame>0)
        {
            var data = value.GetData<uint>();
            Unity.Collections.NativeArray<uint>.Copy(data, Late, data.Length);
        }
        lateCompleted = true;
    }
    public bool TryComplete(bool wait)
    {
        if (!pending)
            return true;
        if (!completed || !lateCompleted)
        {
            if (wait)
            {
                request.WaitForCompletion();
                lateRequest.WaitForCompletion();
            }
            else
            {
                request.Update();
                lateRequest.Update();
            }
        }
        if (!completed || !lateCompleted)
            return false;
        pending = false;
        if (failed)
            throw new InvalidOperationException("Ray response readback failed.");
        return true;
    }
    public void Dispose()
    {
        if (pending && !completed)
            request.WaitForCompletion();
        if (pending && !lateCompleted)
            lateRequest.WaitForCompletion();
        pending = false;
        lateHistogram?.Dispose();lateMaterials?.Dispose();lateSites?.Dispose();lateSiteRanges?.Dispose();
        materialBuffer?.Dispose();
        edgeBuffer?.Dispose();
        volumeBuffer?.Dispose();
        pathBuffer?.Dispose();
        enabledBuffer?.Dispose();
        edgeRangeBuffer?.Dispose();
        edgeIndexBuffer?.Dispose();
        emissionRanges?.Dispose();
        emissionPoints?.Dispose();
        emissionClocks?.Dispose();pairRanges?.Dispose();directSites?.Dispose();
        completeCandidates?.Dispose();completeSelected?.Dispose();completeCount?.Dispose();completeDraws?.Dispose();completeSteps?.Dispose();
        commands?.Dispose();
    }
}
