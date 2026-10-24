using System;
using System.Collections.Generic;
using UnityEngine;

// Shared linear vibration state. Output is metres/metres per second, not pressure or PCM.
// Radiation/propagation must consume this motion through a separately validated adapter.
public sealed class ModalBody
{
    [Serializable] public sealed class Data
    {
        public int version;
        public string quantity, bankSha256;
        public double loss;
        public double[] frequencies, positions, shapes;
        public int[] triangles;
    }

    public sealed class Bank
    {
        readonly Data data;
        public int ModeCount => data.frequencies.Length;
        public int TriangleCount => data.triangles.Length/3;
        public string Hash => data.bankSha256;
        public double MaximumFrequency => data.frequencies[ModeCount-1];

        public Bank(string json)
        {
            data=JsonUtility.FromJson<Data>(json);
            if (data==null || data.version!=1 || data.quantity!="mechanical-surface-motion" ||
                !(data.loss>=0 && data.loss<2) || data.frequencies==null || data.frequencies.Length==0 ||
                data.positions==null || data.positions.Length<9 || data.positions.Length%3!=0 ||
                data.triangles==null || data.triangles.Length==0 || data.triangles.Length%3!=0 ||
                data.shapes==null || data.shapes.LongLength!=(long)data.frequencies.Length*data.positions.Length)
                throw new ArgumentException("Invalid modal bank layout.");
            double last=0;
            foreach(double hz in data.frequencies)
            { if (!(hz>0) || !double.IsFinite(hz) || hz<last) throw new ArgumentException("Invalid mode frequencies."); last=hz; }
            foreach(double value in data.positions) if(!double.IsFinite(value)) throw new ArgumentException("Invalid surface position.");
            foreach(double value in data.shapes) if(!double.IsFinite(value)) throw new ArgumentException("Invalid mode shape.");
            foreach(int vertex in data.triangles) if(vertex<0 || vertex>=data.positions.Length/3) throw new ArgumentException("Invalid surface triangle.");
        }

        public void Couple(int triangle, Vector3 barycentric, Vector3 vector, double[] result)
        {
            if (triangle<0 || triangle>=TriangleCount || result==null || result.Length!=ModeCount ||
                !(barycentric.x>=0 && barycentric.y>=0 && barycentric.z>=0) ||
                Math.Abs((double)barycentric.x+barycentric.y+barycentric.z-1)>1e-6 ||
                !float.IsFinite(vector.x) || !float.IsFinite(vector.y) || !float.IsFinite(vector.z))
                throw new ArgumentException("Invalid surface coupling.");
            for(int mode=0;mode<ModeCount;mode++)
            {
                double value=0;
                for(int corner=0;corner<3;corner++)
                {
                    int index=mode*data.positions.Length+3*data.triangles[3*triangle+corner];
                    value+=barycentric[corner]*(data.shapes[index]*vector.x+data.shapes[index+1]*vector.y+data.shapes[index+2]*vector.z);
                }
                result[mode]=value;
            }
        }

        internal double Frequency(int index) => data.frequencies[index];
        internal double Loss => data.loss;
    }

    public struct Impact
    {
        public long sample;
        public int triangle;
        public Vector3 barycentric, impulse; // N s, expressed in the bank's coordinates.
    }

    readonly Bank bank;
    readonly double[] q, v, a, b, c, d, scratch;
    public long Sample { get; private set; }

    public ModalBody(Bank bank, int sampleRate)
    {
        if (bank==null || sampleRate<=0 || bank.MaximumFrequency>=sampleRate*.5)
            throw new ArgumentException("Retained modes must be below Nyquist.");
        this.bank=bank; int count=bank.ModeCount;
        q=new double[count];v=new double[count];a=new double[count];b=new double[count];c=new double[count];d=new double[count];scratch=new double[count];
        for(int i=0;i<count;i++)
        {
            double omega=2*Math.PI*bank.Frequency(i), alpha=bank.Loss*omega*.5;
            double beta=Math.Sqrt(omega*omega-alpha*alpha), e=Math.Exp(-alpha/sampleRate);
            double cos=Math.Cos(beta/sampleRate), sin=Math.Sin(beta/sampleRate)/beta;
            a[i]=e*(cos+alpha*sin);b[i]=e*sin;c[i]=-e*omega*omega*sin;d[i]=e*(cos-alpha*sin);
        }
    }

    public void Render(int count, IReadOnlyList<Impact> impacts, double[] observation, double[] displacement, double[] velocity)
    {
        long end=checked(Sample+count);
        if(count<0 || impacts==null || observation==null || observation.Length!=q.Length ||
            displacement==null || velocity==null || displacement.Length<count || velocity.Length<count ||
            ReferenceEquals(displacement,velocity) || ReferenceEquals(observation,displacement) || ReferenceEquals(observation,velocity))
            throw new ArgumentException("Invalid modal output buffers.");
        foreach(double value in observation) if(!double.IsFinite(value)) throw new ArgumentException("Invalid observation coupling.");
        long previous=Sample;
        for(int i=0;i<impacts.Count;i++)
        {
            var impact=impacts[i];
            if(impact.sample<previous || impact.sample>=end) throw new ArgumentException("Sorted impacts must belong to this batch.");
            bank.Couple(impact.triangle,impact.barycentric,impact.impulse,scratch);previous=impact.sample;
        }
        int next=0;
        for(int sample=0;sample<count;sample++)
        {
            while(next<impacts.Count && impacts[next].sample==Sample)
            {
                var impact=impacts[next++];bank.Couple(impact.triangle,impact.barycentric,impact.impulse,scratch);
                for(int mode=0;mode<v.Length;mode++)v[mode]+=scratch[mode];
            }
            double position=0,speed=0;
            for(int mode=0;mode<q.Length;mode++)
            {
                position+=observation[mode]*q[mode];speed+=observation[mode]*v[mode];
                double old=q[mode];q[mode]=a[mode]*old+b[mode]*v[mode];v[mode]=c[mode]*old+d[mode]*v[mode];
            }
            displacement[sample]=position;velocity[sample]=speed;Sample++;
        }
    }
}
