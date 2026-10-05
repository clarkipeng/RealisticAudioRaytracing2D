using System;
using UnityEngine;

// A distance-parameterized oval. The Editor track mesh and both vehicles use this same curve.
public sealed class StationLoop : MonoBehaviour
{
    [Min(1)] public float straightLength=56;
    [Min(1)] public float radius=16.5f;
    public double Length => 2*straightLength+2*Math.PI*radius;

    public Pose LocalPose(double distance)
    {
        if(!double.IsFinite(distance) || !float.IsFinite(straightLength) || !float.IsFinite(radius) || straightLength<=0 || radius<=0)
            throw new ArgumentException("Route dimensions and distance must be finite; dimensions must be positive.");
        double d=(distance%Length+Length)%Length, arc=Math.PI*radius;
        Vector3 p; float yaw;
        if(d<straightLength) {p=new Vector3(radius,0,(float)d-straightLength/2); yaw=0;}
        else if((d-=straightLength)<arc)
        {
            double a=d/radius; p=new Vector3((float)(radius*Math.Cos(a)),0,straightLength/2+(float)(radius*Math.Sin(a))); yaw=-(float)(a*Mathf.Rad2Deg);
        }
        else if((d-=arc)<straightLength) {p=new Vector3(-radius,0,straightLength/2-(float)d); yaw=180;}
        else
        {
            double a=(d-straightLength)/radius; p=new Vector3(-(float)(radius*Math.Cos(a)),0,-straightLength/2-(float)(radius*Math.Sin(a))); yaw=180-(float)(a*Mathf.Rad2Deg);
        }
        return new Pose(p,Quaternion.Euler(0,yaw,0));
    }
    public Pose PoseAt(double distance)
    {
        if((transform.lossyScale-Vector3.one).sqrMagnitude>1e-8f) throw new InvalidOperationException("Keep the route at unit scale; edit its length and radius instead.");
        var pose=LocalPose(distance); return new Pose(transform.TransformPoint(pose.position),transform.rotation*pose.rotation);
    }
    public void Place(Transform vehicle,double distance)
    {
        var pose=PoseAt(distance); vehicle.SetPositionAndRotation(pose.position,pose.rotation);
    }
    // Integrate the speed ramp, including a frame that reaches the requested speed part way through.
    public static double Advance(ref float speed,float target,float acceleration,float seconds)
    {
        if(acceleration<=0 || seconds<0 || !float.IsFinite(acceleration) || !float.IsFinite(seconds) || !float.IsFinite(target) || target<0)
            throw new ArgumentException("Train speed/time must be nonnegative and acceleration positive and finite.");
        float ramp=Mathf.Min(seconds,Mathf.Abs(target-speed)/acceleration);
        float next=Mathf.MoveTowards(speed,target,acceleration*seconds);
        double distance=(speed+next)*.5*ramp+next*(seconds-ramp); speed=next; return distance;
    }
    // Integrate acceleration, cruise and braking at their exact transition times.
    // Unlike chasing sqrt(2*a*distance) once per frame, this reaches the platform at zero speed.
    public static double AdvanceToStop(ref float speed,double remaining,float cruise,float acceleration,float seconds)
    {
        if(!double.IsFinite(remaining) || remaining<0) throw new ArgumentException("Stop distance must be finite and nonnegative.");
        Advance(ref speed,speed,acceleration,0); // Shared finite/range validation.
        if(!float.IsFinite(cruise) || cruise<0 || !float.IsFinite(seconds) || seconds<0) throw new ArgumentException("Cruise and time must be finite and nonnegative.");
        double v=speed,a=acceleration,t=seconds,d=remaining;
        if(d<=1e-6) {speed=0; return d;}
        if(v>cruise)
        {
            double dt=Math.Min(t,(v-cruise)/a); d-=v*dt-.5*a*dt*dt; v-=a*dt; t-=dt;
        }
        double peak=Math.Min(cruise,Math.Sqrt(Math.Max(0,a*d+.5*v*v)));
        if(peak>v)
        {
            double dt=Math.Min(t,(peak-v)/a); d-=v*dt+.5*a*dt*dt; v+=a*dt; t-=dt;
        }
        if(v>0)
        {
            double dt=Math.Min(t,Math.Max(0,(d-v*v/(2*a))/v)); d-=v*dt; t-=dt;
            dt=Math.Min(t,v/a); d-=v*dt-.5*a*dt*dt; v-=a*dt;
        }
        if(d<=1e-6) {speed=0; return remaining;}
        speed=(float)v; return remaining-d;
    }
    void OnDrawGizmosSelected()
    {
        if(straightLength<=0 || radius<=0) return;
        Gizmos.color=Color.cyan; Gizmos.matrix=transform.localToWorldMatrix;
        var previous=LocalPose(0).position;
        int steps=Mathf.CeilToInt((float)Length);
        for(int i=1;i<=steps;i++) {var next=LocalPose(Length*i/steps).position; Gizmos.DrawLine(previous,next); previous=next;}
        Gizmos.matrix=Matrix4x4.identity;
    }
}
