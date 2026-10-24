using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

public class ModalBakeChecks
{
    static ModalBody.Bank Bank()
    {
        return new ModalBody.Bank(JsonUtility.ToJson(new ModalBody.Data {version=1,quantity="mechanical-surface-motion",loss=.02,
            frequencies=new double[]{400,730},positions=new double[]{0,0,0,1,0,0,0,1,0},triangles=new[]{0,1,2},
            shapes=new double[]{0,0,2,0,0,1,0,0,-1, 0,0,-1,0,0,3,0,0,2}}));
    }

    [Test, Category("Core")]
    public void SharedModalStateMatchesAnalyticalImpulseAndArbitraryBatchPartitions()
    {
        var bank=Bank();const int count=2000,rate=48000;
        var observation=new double[2];bank.Couple(0,new Vector3(1,0,0),Vector3.forward,observation);
        var impact=new ModalBody.Impact {sample=13,triangle=0,barycentric=new Vector3(1,0,0),impulse=Vector3.forward*.001f};
        var reference=new ModalBody(bank,rate);var position=new double[count];var velocity=new double[count];
        reference.Render(count,new[]{impact},observation,position,velocity);
        for(int sample=13;sample<count;sample++)
        {
            double time=(sample-13)/(double)rate,expected=0;
            for(int mode=0;mode<2;mode++)
            {
                double omega=2*Math.PI*(mode==0 ? 400:730),alpha=.01*omega,beta=Math.Sqrt(omega*omega-alpha*alpha);
                expected+=observation[mode]*observation[mode]*(double)impact.impulse.z*Math.Exp(-alpha*time)*Math.Sin(beta*time)/beta;
            }
            Assert.That(position[sample],Is.EqualTo(expected).Within(1e-15));
        }
        foreach(int batch in new[]{1,17,128,1000})
        {
            var state=new ModalBody(bank,rate);var p=new double[batch];var v=new double[batch];
            while(state.Sample<count)
            {
                int start=(int)state.Sample,n=Math.Min(batch,count-start);
                var events=new List<ModalBody.Impact>();if(impact.sample>=start && impact.sample<start+n)events.Add(impact);
                state.Render(n,events,observation,p,v);
                for(int i=0;i<n;i++) {Assert.That(p[i],Is.EqualTo(position[start+i]));Assert.That(v[i],Is.EqualTo(velocity[start+i]));}
            }
        }
    }

    [Test, Category("Core")]
    public void ModalStateRejectsUnrepresentedEventsAndAliasedOutputs()
    {
        var bank=Bank();var state=new ModalBody(bank,48000);var output=new double[10];
        Assert.Throws<ArgumentException>(()=>state.Render(10,Array.Empty<ModalBody.Impact>(),new double[2],output,output));
        var impact=new ModalBody.Impact {sample=10,triangle=0,barycentric=new Vector3(1,0,0)};
        Assert.Throws<ArgumentException>(()=>state.Render(10,new[]{impact},new double[2],output,new double[10]));
        Assert.That(state.Sample,Is.Zero);
        Assert.Throws<ArgumentException>(()=>new ModalBody(bank,1000));
    }

    [Test, Category("Core")]
    public void ExportPreservesScaledGeometryAndPhysicalMaterialWithoutMutatingObject()
    {
        var target=GameObject.CreatePrimitive(PrimitiveType.Cube);
        var material=ScriptableObject.CreateInstance<AudioMaterial>();
        try
        {
            target.AddComponent<AcousticSurface>().material=material;
            target.transform.position=new Vector3(10,20,30);
            target.transform.localScale=new Vector3(2,3,.2f);
            target.transform.rotation=Quaternion.Euler(10,20,30);
            material.modalYoungModulus=70e9f; material.density=2700;
            var box=ModalBakeEditor.Describe(target,true);
            Assert.That(box.geometry.size,Is.EqualTo(new double[]{2,3,.2}).Within(1e-6));
            Assert.That(box.material.young,Is.EqualTo((double)material.modalYoungModulus));
            Assert.That(box.material.density,Is.EqualTo(2700));
            Assert.That(box.support,Is.EqualTo("free"));
            var mesh=ModalBakeEditor.Describe(target,false);
            var vertex=target.transform.TransformVector(target.GetComponent<MeshFilter>().sharedMesh.vertices[0]);
            Assert.That(mesh.geometry.vertices[0],Is.EqualTo(vertex.x));
            Assert.That(mesh.geometry.vertices[1],Is.EqualTo(vertex.y));
            Assert.That(mesh.geometry.vertices[2],Is.EqualTo(vertex.z));
            Assert.That(target.transform.position,Is.EqualTo(new Vector3(10,20,30)));
            Assert.That(JsonUtility.FromJson<ModalBakeEditor.Input>(JsonUtility.ToJson(mesh)).geometry.triangles.Length,Is.EqualTo(36));
        }
        finally { UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(material); }
    }

    [Test, Category("Core")]
    public void ExportRejectsMixedMaterialsAndInvalidSolidProperties()
    {
        var target=new GameObject("solid"); var material=ScriptableObject.CreateInstance<AudioMaterial>();
        var other=ScriptableObject.CreateInstance<AudioMaterial>();
        try
        {
            var surface=target.AddComponent<AcousticSurface>(); surface.material=material;
            target.AddComponent<BoxCollider>(); surface.submeshMaterials=new[]{other};
            Assert.Throws<ArgumentException>(()=>ModalBakeEditor.Describe(target,true));
            surface.submeshMaterials=null; material.modalPoissonRatio=.5f;
            Assert.Throws<ArgumentException>(()=>ModalBakeEditor.Describe(target,true));
        }
        finally { UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(material); UnityEngine.Object.DestroyImmediate(other); }
    }
}
