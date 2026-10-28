using UnityEngine;
using System.Collections.Generic;
using System;


public static class SceneToTake
{
    public static AcousticTake.State Observe(AcousticTake.Scene scene, int index, Transform transform)
    {
        var flags = scene.objects[index].primitiveCount > 0 ? AcousticTake.StateFlags.Geometry : 0;
        var surface = transform.GetComponent<AcousticSurface>();
        if (surface != null && !surface.enabled) flags &= ~AcousticTake.StateFlags.Geometry;
        for (int s = 0; s < scene.sources.Length; s++) if (scene.sources[s].objectIndex == index)
        {
            var component = transform.GetComponent<RaytracedAudioSource>();
            if (component == null || component.enabled) flags |= AcousticTake.StateFlags.Source;
        }
        if (Array.IndexOf(scene.listeners, index) >= 0) flags |= AcousticTake.StateFlags.Listener;
        var state = AcousticTake.State.Observe(transform, flags);
        if (scene.dimension == AcousticTake.Dimension.Planar2D &&
            (Mathf.Abs(state.position.z) > .0001f || Mathf.Abs(Vector3.Dot(transform.forward, Vector3.forward)) < .99999f))
            throw new InvalidOperationException("Planar capture uses the XY plane; use a 3D take for out-of-plane objects.");
        return state;
    }
    public static AcousticTake.Scene Compile(List<RayTraceManager.TakeObjectBinding> bindings,
        AcousticTake.Dimension dimension, float curveTolerance, AudioMaterial fallback,
        Transform[] sources, AudioClip[] clips, Transform[] listeners,
        out AcousticTake.Primitive[] geometry, bool authoringCoordinates = false,
        List<AudioMaterial> materialAssets = null, bool useSolidBakes = true)
    {
        if (!(curveTolerance > 0)) throw new ArgumentOutOfRangeException(nameof(curveTolerance));
        var primitives = new List<AcousticTake.Primitive>();
        var materials = new List<AcousticTake.MaterialDefinition>();
        var materialIndices = new Dictionary<AudioMaterial, int>();
        var objects = new AcousticTake.ObjectDefinition[bindings.Count];
        var objectIndices = new Dictionary<Transform, int>();
        for (int i = 0; i < bindings.Count; i++) objectIndices.Add(bindings[i].transform, i);
        var grouped = new HashSet<Transform>();
        var boundaries = new Dictionary<Transform, AcousticSolid.Boundary>();
        if (useSolidBakes) foreach (var binding in bindings)
        {
            var solid = binding.transform.GetComponent<AcousticSolid>();
            if (solid == null || !solid.enabled || !binding.geometry) continue;
            if (binding.transform.GetComponent<AcousticSurface>().geometrySource == AcousticSurface.GeometrySource.None) continue;
            if (dimension != AcousticTake.Dimension.Spatial3D) throw new InvalidOperationException("Acoustic solids require 3D capture.");
            boundaries.Add(binding.transform, solid.Resolve(fallback));
            foreach (var member in solid.members)
            {
                if (!objectIndices.TryGetValue(member.transform, out int at) || !bindings[at].geometry)
                    throw new InvalidOperationException("Register each acoustic solid member as a geometry binding.");
                if (!grouped.Add(member.transform)) throw new InvalidOperationException("A surface cannot belong to two acoustic solids.");
            }
        }

        for (int i = 0; i < bindings.Count; i++)
        {
            var binding = bindings[i];
            var transform = binding.transform;
            int start = primitives.Count;
            var surface = transform.GetComponent<AcousticSurface>();
            var selection = surface != null ? surface.geometrySource : AcousticSurface.GeometrySource.Auto;
            var solid = useSolidBakes ? transform.GetComponent<AcousticSolid>() : null;
            var filter = transform.GetComponent<MeshFilter>();
            var inverseRotation = Quaternion.Inverse(transform.rotation);
            Func<Vector3, Vector3> bake = p => authoringCoordinates ? p : inverseRotation * transform.TransformVector(p);
            Func<int, int> material = slot =>
            {
                var asset = surface != null && surface.material != null ? surface.material : fallback;
                if (surface != null && surface.submeshMaterials != null && slot < surface.submeshMaterials.Length && surface.submeshMaterials[slot] != null)
                    asset = surface.submeshMaterials[slot];
                if (solid != null && solid.enabled) asset = solid.SurfaceMaterials[slot];
                if (asset == null) throw new InvalidOperationException(transform.name + ": assign an acoustic material or the take's default material.");
                if (!materialIndices.TryGetValue(asset, out int index))
                {
                    index = materials.Count; materialIndices.Add(asset, index); materialAssets?.Add(asset);
                    materials.Add(asset.ExportEnergyCurve());
                }
                return index;
            };
            Action<Vector3, Vector3, Vector3, int> add = (a, b, c, slot) =>
            {
                a = bake(a); b = bake(b); c = bake(c);
                if ((a - b).sqrMagnitude < 1e-12f || (dimension == AcousticTake.Dimension.Spatial3D && Vector3.Cross(b-a, c-a).sqrMagnitude < 1e-16f))
                    throw new InvalidOperationException(transform.name + ": degenerate acoustic geometry.");
                primitives.Add(new AcousticTake.Primitive { a = a, b = b, c = c, objectIndex = i, materialIndex = material(slot) });
            };
            if (binding.geometry && !grouped.Contains(transform) && selection != AcousticSurface.GeometrySource.None)
            {
                if (solid != null)
                {
                    if (solid.enabled)
                    {
                        var data = boundaries[transform];
                        Vector3 Point(int id) => new Vector3(data.vertices[3*id], data.vertices[3*id+1], data.vertices[3*id+2]);
                        for (int t = 0; t < data.materials.Length; t++)
                            add(Point(data.triangles[3*t]), Point(data.triangles[3*t+1]), Point(data.triangles[3*t+2]), data.materials[t]);
                    }
                }
                else if (dimension == AcousticTake.Dimension.Planar2D)
                    Segments(transform, curveTolerance, add);
                else if (selection == AcousticSurface.GeometrySource.Mesh || (selection == AcousticSurface.GeometrySource.Auto && filter != null))
                    Mesh(filter != null ? filter.sharedMesh : null, add);
                else
                    ColliderGeometry(transform, curveTolerance, add);
            }
            objects[i] = new AcousticTake.ObjectDefinition { id = binding.id, name = transform.name,
                primitiveStart = start, primitiveCount = primitives.Count - start, scale = authoringCoordinates ? Vector3.one : transform.lossyScale };
        }
        var sourceDefinitions = new AcousticTake.SourceDefinition[sources.Length];
        var planes = new List<AcousticTake.EmissionPlane>();
        var clipDefinitions = new List<AcousticTake.ClipDefinition>();
        var clipIndices = new Dictionary<AudioClip,int>();
        for (int i = 0; i < sources.Length; i++)
        {
            var source=sources[i].GetComponent<RaytracedAudioSource>();
            var plane=sources[i].GetComponent<AudioSourcePlane>();
            if(plane!=null && plane.enabled)planes.Add(plane.Export(i,dimension));
            if(clips[i]==null)
            {
                sourceDefinitions[i]=new AcousticTake.SourceDefinition {objectIndex=objectIndices[sources[i]],clipIndex=-1,streamed=true,mutedListenerPlusOne=source==null?0:source.mutedListenerPlusOne};
                continue;
            }
            if (!clipIndices.TryGetValue(clips[i],out int clipIndex))
            {
                clipIndex=clipDefinitions.Count; clipIndices.Add(clips[i],clipIndex);
                clipDefinitions.Add(new AcousticTake.ClipDefinition {name=clips[i].name});
            }
            sourceDefinitions[i] = new AcousticTake.SourceDefinition { objectIndex = objectIndices[sources[i]], clipIndex = clipIndex };
        }
        var listenerIndices = new int[listeners.Length];
        for (int i = 0; i < listeners.Length; i++) listenerIndices[i] = objectIndices[listeners[i]];
        geometry = primitives.ToArray();
        return new AcousticTake.Scene { dimension = dimension, objects = objects, sources = sourceDefinitions,
            listeners = listenerIndices, clips = clipDefinitions.ToArray(), materials = materials.ToArray(), planes = planes.ToArray() };
    }

    // Compare authored inputs without re-tessellating shapes or rebuilding material tables.
    // Mesh data is still read: Unity provides no general notification for in-place mesh edits.
    public sealed class Guard
    {
        readonly RayTraceManager manager;
        readonly Transform root;
        readonly AcousticTake.Dimension dimension;
        readonly float tolerance;
        readonly AudioMaterial fallback;
        readonly RaytracedAudioSource[] sourceBindings;
        readonly int[] sourceComponentIds;
        readonly Transform[] sourceObjects;
        readonly AudioClip[] sourceClips;
        readonly bool[] streamed;
        readonly int[] muted;
        readonly PlaneSnapshot[] planes;
        readonly Transform[] listenerBindings;
        readonly BindingSnapshot[] bindings;
        readonly List<AcousticSurface> surfaces=new List<AcousticSurface>(), surfaceScratch=new List<AcousticSurface>();
        readonly List<RaytracedAudioSource> emitters=new List<RaytracedAudioSource>(), emitterScratch=new List<RaytracedAudioSource>();
        readonly Dictionary<Mesh,MeshSnapshot> meshes=new Dictionary<Mesh,MeshSnapshot>();
        readonly Dictionary<AudioMaterial,MaterialSnapshot> materials=new Dictionary<AudioMaterial,MaterialSnapshot>();
        public Guard(RayTraceManager manager,Transform[] sources,AudioClip[] clips,Transform[] listeners)
        {
            this.manager=manager; root=manager.takeRoot;
            dimension=manager.takeDimension; tolerance=manager.takeCurveTolerance; fallback=manager.takeDefaultMaterial;
            sourceBindings=manager.takeSources.ToArray(); listenerBindings=manager.takeListeners.ToArray();
            streamed=Array.ConvertAll(sources,t=>t.GetComponent<RaytracedAudioSource>()?.streamed??false);
            muted=Array.ConvertAll(sources,t=>t.GetComponent<RaytracedAudioSource>()?.mutedListenerPlusOne??0);
            planes=Array.ConvertAll(sources,t=>new PlaneSnapshot(t));
            sourceObjects=sources; sourceClips=clips; sourceComponentIds=new int[sources.Length];
            for (int i=0;i<sources.Length;i++) sourceComponentIds[i]=Identity(sources[i].GetComponent<RaytracedAudioSource>());
            bindings=new BindingSnapshot[manager.takeObjects.Count];
            for (int i=0;i<bindings.Length;i++) bindings[i]=new BindingSnapshot(manager.takeObjects[i],dimension,this);
            AddMaterial(fallback);
            if (root!=null) { root.GetComponentsInChildren(true,surfaces); root.GetComponentsInChildren(true,emitters); }
            Validate(); // Warm comparison buffers and generic comparers before the sample clock starts.
        }
        void AddMaterial(AudioMaterial value) { if (value!=null && !materials.ContainsKey(value)) materials.Add(value,new MaterialSnapshot(value)); }
        void AddMesh(Mesh value) { if (value!=null && !meshes.ContainsKey(value)) meshes.Add(value,new MeshSnapshot(value)); }
        static readonly Unity.Profiling.ProfilerMarker validationMarker=new Unity.Profiling.ProfilerMarker("Acoustics.ValidateScene");
        public void Validate()
        {
            using (validationMarker.Auto())
            {
                if (manager.takeRoot!=root ||
                    dimension!=manager.takeDimension || tolerance!=manager.takeCurveTolerance || fallback!=manager.takeDefaultMaterial ||
                    !Same(sourceBindings,manager.takeSources) || !Same(listenerBindings,manager.takeListeners) || bindings.Length!=manager.takeObjects.Count)
                    Changed();
                for (int i=0;i<bindings.Length;i++) bindings[i].Validate(manager.takeObjects[i]);
                for (int i=0;i<sourceObjects.Length;i++)
                {
                    planes[i].Validate(sourceObjects[i]);
                    var component=sourceObjects[i].GetComponent<RaytracedAudioSource>();
                    if(component!=null && (component.streamed!=streamed[i] || component.mutedListenerPlusOne!=muted[i]))Changed();
                    if (Identity(component)!=sourceComponentIds[i] || (component==null || (component.streamed?null:component.clip)!=sourceClips[i])) Changed();
                }
                if (root!=null)
                {
                    root.GetComponentsInChildren(true,surfaceScratch); root.GetComponentsInChildren(true,emitterScratch);
                    if (!Same(surfaces,surfaceScratch) || !Same(emitters,emitterScratch)) Changed();
                }
                foreach (var item in meshes) item.Value.Validate(item.Key);
                foreach (var item in materials) item.Value.Validate(item.Key);
            }
        }
        // Missing components can have different managed null wrappers in the Editor.
        // Snapshot native identity so absence is stable and removal/replacement still fails.
        static int Identity(UnityEngine.Object value) => value==null ? 0:value.GetInstanceID();
        static void Changed() { throw new InvalidOperationException("Capture geometry/material/membership/configuration changed; start a new take."); }
        static bool Same<T>(IList<T> a,IList<T> b)
        {
            if (a==null || b==null) return a==b;
            if (a.Count!=b.Count) return false;
            var comparer=EqualityComparer<T>.Default;
            for (int i=0;i<a.Count;i++) if (!comparer.Equals(a[i],b[i])) return false;
            return true;
        }
        static bool Near(Vector3 a,Vector3 b) => (a-b).sqrMagnitude<=1e-10f*Mathf.Max(1,b.sqrMagnitude);
        sealed class PlaneSnapshot
        {
            readonly AudioSourcePlane plane;
            readonly Texture2D mask;
            readonly uint updates;
            readonly Vector2 size;
            readonly int samples,seed,id,period,duration;
            readonly bool enabled,oneShot,project;
            public PlaneSnapshot(Transform source)
            {
                plane=source.GetComponent<AudioSourcePlane>();id=Identity(plane);
                if(plane==null)return;
                mask=plane.mask;updates=mask==null?0:mask.updateCount;size=plane.size;
                samples=plane.samples;seed=plane.seed;enabled=plane.enabled;period=plane.burstPeriodSamples;duration=plane.burstDurationSamples;
                oneShot=plane.oneShotImpacts;project=plane.projectImpacts;
            }
            public void Validate(Transform source)
            {
                if(Identity(source.GetComponent<AudioSourcePlane>())!=id)Changed();
                if(plane!=null && (plane.enabled!=enabled || plane.size!=size || plane.samples!=samples || plane.seed!=seed ||
                    plane.burstPeriodSamples!=period || plane.burstDurationSamples!=duration || plane.oneShotImpacts!=oneShot || plane.projectImpacts!=project || plane.mask!=mask || (mask!=null && mask.updateCount!=updates)))Changed();
            }
        }
        sealed class BindingSnapshot
        {
            readonly Transform transform,parent;
            readonly string id;
            readonly bool geometry;
            readonly Vector3 x,y,z;
            readonly AcousticSurface surface;
            readonly AcousticSurface.GeometrySource selection;
            readonly AudioMaterial material;
            readonly AudioMaterial[] submaterials;
            readonly MeshFilter filter;
            readonly int surfaceId,filterId;
            readonly Mesh mesh;
            readonly AcousticSolid solid; readonly int solidId; readonly AcousticSolid.Snapshot solidSnapshot;
            readonly Component collider;
            readonly List<Collider> colliders=new List<Collider>();
            readonly List<Collider2D> colliders2D=new List<Collider2D>();
            readonly List<Vector2> points=new List<Vector2>();
            readonly Vector2[][] paths;
            readonly Vector3 shapeA,shapeB;
            readonly int axis;
            readonly AcousticTake.Dimension dimension;
            public BindingSnapshot(RayTraceManager.TakeObjectBinding binding,AcousticTake.Dimension dimension,Guard owner)
            {
                this.dimension=dimension; transform=binding.transform; parent=transform.parent; id=binding.id; geometry=binding.geometry;
                var q=Quaternion.Inverse(transform.rotation); x=q*transform.TransformVector(Vector3.right); y=q*transform.TransformVector(Vector3.up); z=q*transform.TransformVector(Vector3.forward);
                surface=transform.GetComponent<AcousticSurface>(); surfaceId=Identity(surface); selection=surface!=null ? surface.geometrySource:AcousticSurface.GeometrySource.Auto;
                material=surface!=null ? surface.material:null; submaterials=surface!=null && surface.submeshMaterials!=null ? (AudioMaterial[])surface.submeshMaterials.Clone():null;
                owner.AddMaterial(material); if (submaterials!=null) foreach (var value in submaterials) owner.AddMaterial(value);
                solid=transform.GetComponent<AcousticSolid>(); solidId=Identity(solid);
                if(solid!=null)
                {
                    solidSnapshot=new AcousticSolid.Snapshot(solid,geometry && selection!=AcousticSurface.GeometrySource.None);
                    foreach(var value in solid.SurfaceMaterials)owner.AddMaterial(value);
                    return;
                }
                if (!geometry || selection==AcousticSurface.GeometrySource.None) return;
                filter=transform.GetComponent<MeshFilter>(); filterId=Identity(filter);
                if (dimension==AcousticTake.Dimension.Spatial3D && (selection==AcousticSurface.GeometrySource.Mesh || (selection==AcousticSurface.GeometrySource.Auto && filter!=null))) mesh=filter.sharedMesh;
                else
                {
                    if (dimension==AcousticTake.Dimension.Planar2D) { transform.GetComponents(colliders2D); collider=colliders2D[0]; }
                    else { transform.GetComponents(colliders); collider=colliders[0]; }
                    if (collider is MeshCollider mc) mesh=mc.sharedMesh;
                    Shape(out shapeA,out shapeB,out axis);
                    int count=collider is PolygonCollider2D polygon ? polygon.pathCount:collider is EdgeCollider2D ? 1:0;
                    paths=new Vector2[count][];
                    for (int i=0;i<count;i++) { ReadPath(i); paths[i]=points.ToArray(); }
                }
                owner.AddMesh(mesh);
            }
            void ReadPath(int index)
            {
                points.Clear();
                if (collider is PolygonCollider2D polygon) polygon.GetPath(index,points);
                else ((EdgeCollider2D)collider).GetPoints(points);
            }
            void Shape(out Vector3 a,out Vector3 b,out int direction)
            {
                a=b=Vector3.zero; direction=0;
                if (collider is BoxCollider box) { a=box.center; b=box.size; }
                else if (collider is SphereCollider sphere) { a=sphere.center; b.x=sphere.radius; }
                else if (collider is CapsuleCollider capsule) { a=capsule.center; b=new Vector3(capsule.radius,capsule.height); direction=capsule.direction; }
                else if (collider is Collider2D c)
                {
                    a=c.offset;
                    if (c is BoxCollider2D box2) b=box2.size;
                    else if (c is CircleCollider2D circle) b.x=circle.radius;
                    else if (c is CapsuleCollider2D capsule2) { b=capsule2.size; direction=(int)capsule2.direction; }
                }
            }
            public void Validate(RayTraceManager.TakeObjectBinding binding)
            {
                if (binding==null || transform==null || binding.transform!=transform || binding.id!=id || binding.geometry!=geometry || transform.parent!=parent || Identity(transform.GetComponent<AcousticSurface>())!=surfaceId) Changed();
                var q=Quaternion.Inverse(transform.rotation);
                if (!Near(x,q*transform.TransformVector(Vector3.right)) || !Near(y,q*transform.TransformVector(Vector3.up)) || !Near(z,q*transform.TransformVector(Vector3.forward))) Changed();
                if (surface!=null && (surface.geometrySource!=selection || surface.material!=material || !Same(submaterials,surface.submeshMaterials))) Changed();
                if(Identity(transform.GetComponent<AcousticSolid>())!=solidId)Changed();
                if(solidSnapshot!=null){solidSnapshot.Validate();return;}
                if (!geometry || selection==AcousticSurface.GeometrySource.None) return;
                if (Identity(transform.GetComponent<MeshFilter>())!=filterId) Changed();
                if (ReferenceEquals(collider,null)) { if (filter==null || filter.sharedMesh!=mesh) Changed(); return; }
                if (dimension==AcousticTake.Dimension.Planar2D) { transform.GetComponents(colliders2D); if (colliders2D.Count!=1 || colliders2D[0]!=collider) Changed(); }
                else { transform.GetComponents(colliders); if (colliders.Count!=1 || colliders[0]!=collider) Changed(); }
                if (collider is MeshCollider mc && mc.sharedMesh!=mesh) Changed();
                Shape(out var a,out var b,out var d);
                if (!a.Equals(shapeA) || !b.Equals(shapeB) || d!=axis) Changed();
                if (collider is PolygonCollider2D polygon && polygon.pathCount!=paths.Length) Changed();
                for (int i=0;i<paths.Length;i++) { ReadPath(i); if (!Same(paths[i],points)) Changed(); }
            }
        }
        sealed class MeshSnapshot
        {
            readonly Vector3[] vertices;
            readonly int[][] indices;
            readonly List<Vector3> vertexScratch=new List<Vector3>();
            readonly List<int> indexScratch=new List<int>();
            public MeshSnapshot(Mesh mesh)
            {
                vertices=mesh.vertices; indices=new int[mesh.subMeshCount][];
                for (int i=0;i<indices.Length;i++) indices[i]=mesh.GetTriangles(i);
                Validate(mesh); // Reserve scratch storage before the observation loop.
            }
            public void Validate(Mesh mesh)
            {
                if (mesh==null || !mesh.isReadable || mesh.subMeshCount!=indices.Length) Changed();
                mesh.GetVertices(vertexScratch); if (!Same(vertices,vertexScratch)) Changed();
                for (int i=0;i<indices.Length;i++)
                {
                    if (mesh.GetTopology(i)!=MeshTopology.Triangles) Changed();
                    mesh.GetTriangles(indexScratch,i); if (!Same(indices[i],indexScratch)) Changed();
                }
            }
        }
        sealed class MaterialSnapshot
        {
            readonly AudioMaterial.EnergyKnot[] knots;
            readonly AudioMaterial.AdmittanceKnot[] admittance;
            readonly float roughness,density,porosity,rigidity;
            readonly bool trans,bands;
            readonly float[] absorption,scattering,transmission,ior,damping;
            static T[] Copy<T>(T[] values) => values==null ? null:(T[])values.Clone();
            public MaterialSnapshot(AudioMaterial m)
            {
                knots=Copy(m.energyCurve); admittance=Copy(m.admittanceCurve); roughness=m.roughness; density=m.density; porosity=m.porosity; rigidity=m.rigidity; trans=m.overrideTrans; bands=m.overrideBands;
                absorption=Copy(m.absorptionBands); scattering=Copy(m.scatteringBands); transmission=Copy(m.transmissionBands); ior=Copy(m.iorBands); damping=Copy(m.dampingBands);
            }
            public void Validate(AudioMaterial m)
            {
                if (m==null || roughness!=m.roughness || density!=m.density || porosity!=m.porosity || rigidity!=m.rigidity || trans!=m.overrideTrans || bands!=m.overrideBands ||
                    !Same(absorption,m.absorptionBands) || !Same(scattering,m.scatteringBands) || !Same(transmission,m.transmissionBands) || !Same(ior,m.iorBands) || !Same(damping,m.dampingBands)) Changed();
                // Explicit value comparisons avoid boxing EnergyKnot on the hot path.
                if((admittance==null)!=(m.admittanceCurve==null))Changed();
                if(admittance!=null)
                {
                    if(admittance.Length!=m.admittanceCurve.Length)Changed();
                    for(int i=0;i<admittance.Length;i++)if(admittance[i].frequencyHz!=m.admittanceCurve[i].frequencyHz ||
                        admittance[i].real!=m.admittanceCurve[i].real || admittance[i].imaginary!=m.admittanceCurve[i].imaginary)Changed();
                }
                if (knots==null || m.energyCurve==null) { if (knots!=m.energyCurve) Changed(); return; }
                if (knots.Length!=m.energyCurve.Length) Changed();
                for (int i=0;i<knots.Length;i++) if (knots[i].frequencyHz!=m.energyCurve[i].frequencyHz || knots[i].absorption!=m.energyCurve[i].absorption || knots[i].scattering!=m.energyCurve[i].scattering) Changed();
            }
        }
    }

    static void Mesh(Mesh mesh, Action<Vector3, Vector3, Vector3, int> add)
    {
        if (mesh == null || !mesh.isReadable) throw new InvalidOperationException("Acoustic mesh is missing or not readable.");
        var vertices = mesh.vertices;
        for (int slot = 0; slot < mesh.subMeshCount; slot++)
        {
            if (mesh.GetTopology(slot) != MeshTopology.Triangles) throw new InvalidOperationException("Acoustic meshes require triangle topology.");
            var indices = mesh.GetTriangles(slot);
            for (int t = 0; t < indices.Length; t += 3) add(vertices[indices[t]], vertices[indices[t+1]], vertices[indices[t+2]], slot);
        }
    }

    static void ColliderGeometry(Transform transform, float tolerance, Action<Vector3, Vector3, Vector3, int> add)
    {
        var colliders = transform.GetComponents<Collider>();
        if (colliders.Length != 1) throw new InvalidOperationException(transform.name + ": select a mesh/proxy or use one supported collider.");
        if (colliders[0] is BoxCollider box)
        {
            var v = new Vector3[8];
            for (int i = 0; i < 8; i++) v[i] = box.center + Vector3.Scale(box.size * .5f,
                new Vector3((i&1)==0 ? -1:1, (i&2)==0 ? -1:1, (i&4)==0 ? -1:1));
            int[] triangles = {0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};
            for (int i = 0; i < triangles.Length; i += 3) add(v[triangles[i]], v[triangles[i+1]], v[triangles[i+2]], 0);
        }
        else if (colliders[0] is MeshCollider mesh) Mesh(mesh.sharedMesh, add);
        else if (colliders[0] is SphereCollider || colliders[0] is CapsuleCollider)
        {
            var scale = transform.lossyScale;
            float sx = Mathf.Abs(scale.x), sy = Mathf.Abs(scale.y), sz = Mathf.Abs(scale.z);
            if (!Mathf.Approximately(sx, sy) || !Mathf.Approximately(sx, sz))
                throw new InvalidOperationException(transform.name + ": use a mesh for nonuniformly scaled curved colliders.");
            var sphere = colliders[0] as SphereCollider;
            var capsule = colliders[0] as CapsuleCollider;
            float radius = sphere != null ? sphere.radius : capsule.radius;
            float half = sphere != null ? 0 : Mathf.Max(0, capsule.height * .5f - radius);
            Vector3 center = sphere != null ? sphere.center : capsule.center;
            int axis = sphere != null ? 1 : capsule.direction;
            int slices = CurveSteps(radius * sx, tolerance), hemisphere = Mathf.Max(1, (slices + 3)/4);
            var profile=new List<Vector2>(); // radius from axis, height; both capsule equators are explicit
            for (int r=0; r<=hemisphere; r++)
            {
                float angle=Mathf.PI*.5f*r/hemisphere;
                profile.Add(new Vector2(Mathf.Sin(angle)*radius,half+Mathf.Cos(angle)*radius));
            }
            for (int r=half>0 ? 0:1; r<=hemisphere; r++)
            {
                float angle=Mathf.PI*.5f*r/hemisphere;
                profile.Add(new Vector2(Mathf.Cos(angle)*radius,-half-Mathf.Sin(angle)*radius));
            }
            Func<int, int, Vector3> point = (ring, slice) =>
            {
                float longitude = 2 * Mathf.PI * slice / slices;
                var p = new Vector3(profile[ring].x*Mathf.Cos(longitude),profile[ring].y,profile[ring].x*Mathf.Sin(longitude));
                if (axis == 0) p = new Vector3(p.y, p.x, p.z);
                if (axis == 2) p = new Vector3(p.x, p.z, p.y);
                return center + p;
            };
            for (int r = 0; r < profile.Count-1; r++) for (int s = 0; s < slices; s++)
            {
                var a=point(r,s); var b=point(r,s+1); var c=point(r+1,s); var d=point(r+1,s+1);
                if (r > 0) add(a,b,c,0);
                if (r+1 < profile.Count-1) add(b,d,c,0);
            }
        }
        else throw new InvalidOperationException(transform.name + ": unsupported collider; use a mesh/proxy.");
    }

    static int CurveSteps(float radius, float tolerance)
    {
        if (!(radius > 0)) throw new InvalidOperationException("Curved geometry requires a positive radius.");
        double angle=Math.Acos(Math.Max(-1,1-(double)tolerance/radius));
        if (angle<=0 || Math.PI/angle>int.MaxValue) throw new InvalidOperationException("Curve tolerance is below numeric resolution.");
        return Math.Max(4,(int)Math.Ceiling(Math.PI/angle));
    }

    static void Segments(Transform transform, float tolerance, Action<Vector3, Vector3, Vector3, int> add)
    {
        var colliders = transform.GetComponents<Collider2D>();
        if (colliders.Length != 1) throw new InvalidOperationException(transform.name + ": planar takes require one explicit Collider2D.");
        var col = colliders[0];
        Action<Vector2[], bool> loop = (points, closed) =>
        {
            for (int i=0; i < points.Length - (closed ? 0:1); i++)
                add(points[i]+col.offset, points[(i+1)%points.Length]+col.offset, Vector3.zero, 0);
        };
        if (col is BoxCollider2D box)
        {
            var h = box.size*.5f;
            loop(new[] { new Vector2(-h.x,-h.y), new Vector2(h.x,-h.y), new Vector2(h.x,h.y), new Vector2(-h.x,h.y) }, true);
        }
        else if (col is PolygonCollider2D polygon)
            for (int p=0; p<polygon.pathCount; p++) loop(polygon.GetPath(p), true);
        else if (col is EdgeCollider2D edge) loop(edge.points, false);
        else if (col is CircleCollider2D circle)
        {
            int n=CurveSteps(circle.radius * Mathf.Max(Mathf.Abs(transform.lossyScale.x),Mathf.Abs(transform.lossyScale.y)), tolerance);
            var points = new Vector2[n];
            for (int p=0; p<n; p++) points[p] = new Vector2(Mathf.Cos(2*Mathf.PI*p/n),Mathf.Sin(2*Mathf.PI*p/n))*circle.radius;
            loop(points, true);
        }
        else if (col is CapsuleCollider2D capsule)
        {
            bool horizontal=capsule.direction==CapsuleDirection2D.Horizontal;
            float radius=(horizontal ? capsule.size.y:capsule.size.x)*.5f;
            float half=Mathf.Max(0,(horizontal ? capsule.size.x:capsule.size.y)*.5f-radius);
            int arc=Mathf.Max(2,(CurveSteps(radius*Mathf.Max(Mathf.Abs(transform.lossyScale.x),Mathf.Abs(transform.lossyScale.y)),tolerance)+1)/2);
            var points=new List<Vector2>();
            for (int cap=0; cap<2; cap++) for (int j=0; j<=arc; j++)
            {
                float angle=Mathf.PI*(cap+(float)j/arc);
                var p=new Vector2(radius*Mathf.Cos(angle),radius*Mathf.Sin(angle)+(cap==0 ? half:-half));
                if (horizontal) p=new Vector2(p.y,p.x);
                if (points.Count==0 || (points[points.Count-1]-p).sqrMagnitude>1e-12f) points.Add(p);
            }
            if ((points[0]-points[points.Count-1]).sqrMagnitude<1e-12f) points.RemoveAt(points.Count-1);
            loop(points.ToArray(),true);
        }
        else throw new InvalidOperationException(transform.name + ": unsupported planar collider; use a polygon proxy.");
    }
}
