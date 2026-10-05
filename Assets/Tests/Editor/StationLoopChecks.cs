using System;
using UnityEngine;

public static partial class TakeChecks
{
    static void StationLoopMotion()
    {
        var go=new GameObject("Loop fixture");
        try
        {
            var route=go.AddComponent<StationLoop>(); route.straightLength=20; route.radius=10;
            Near((float)route.Length,(float)(40+20*Math.PI),"Oval has analytic arc length",1e-4f);
            var a=route.LocalPose(0); var b=route.LocalPose(route.Length);
            Near(Vector3.Distance(a.position,b.position),0,"Loop position closes exactly");
            Near(Quaternion.Angle(a.rotation,b.rotation),0,"Loop orientation closes exactly");
            foreach(double boundary in new[]{0.0,20,20+10*Math.PI,40+10*Math.PI})
            {
                var before=route.LocalPose(boundary-.001); var after=route.LocalPose(boundary+.001);
                Check(Vector3.Distance(before.position,after.position)<.0021f,"No position jump at a straight/curve or loop boundary.");
                Check(Quaternion.Angle(before.rotation,after.rotation)<.05f,"Heading is continuous across route boundaries.");
            }
            a=route.LocalPose(20+5*Math.PI);
            Near(Vector3.Distance(a.position,new Vector3(0,0,20)),0,"Top of arc agrees with independent circle geometry",1e-5f);
            Near(Vector3.Distance(a.rotation*Vector3.forward,Vector3.left),0,"Arc heading follows its tangent",1e-5f);
            a=route.LocalPose(20+10*Math.PI+10); Near(Vector3.Distance(a.position,new Vector3(-10,0,0)),0,"Return track travels back beside the platform");
            b=route.LocalPose(-7.2); a=route.LocalPose(route.Length-7.2); Near(Vector3.Distance(a.position,b.position),0,"Carriage wraps behind locomotive without a position cut");
            go.transform.SetPositionAndRotation(new Vector3(5,2,3),Quaternion.Euler(0,30,0));
            a=route.PoseAt(0); Near(Vector3.Distance(a.position,go.transform.position+go.transform.rotation*new Vector3(10,0,-10)),0,"Route placement uses its scene transform",1e-5f);
            float speed=0; double distance=StationLoop.Advance(ref speed,6,2,5);
            Near((float)distance,21,"Acceleration integrates ramp plus cruise distance"); Near(speed,6,"Speed reaches its configured cruise speed");
            float splitSpeed=0; double split=0; for(int i=0;i<50;i++) split+=StationLoop.Advance(ref splitSpeed,6,2,.1f);
            Near((float)split,(float)distance,"Acceleration distance is independent of frame partition",1e-4f);
            distance=StationLoop.Advance(ref speed,0,2,5); Near((float)distance,9,"Braking stops after the expected distance"); Near(speed,0,"Stopped train does not reverse");
            speed=0; distance=StationLoop.AdvanceToStop(ref speed,8,12,2,3);
            Near((float)distance,7,"Short station run uses a triangular speed profile"); Near(speed,2,"Braking begins within a large frame");
            distance+=StationLoop.AdvanceToStop(ref speed,8-distance,12,2,2);
            Near((float)distance,8,"Train stops exactly at the platform"); Near(speed,0,"Station arrival has zero speed");
            foreach(float step in new[]{1f/144,1f/30,.1f,1.7f})
            {
                speed=0; distance=0; double total=route.Length;
                for(int i=0;i<Math.Ceiling(30/step);i++) distance+=StationLoop.AdvanceToStop(ref speed,Math.Max(0,total-distance),12,3,step);
                Near((float)distance,(float)total,"Station stop does not overshoot at different frame rates",1e-4f); Near(speed,0,"Each frame rate reaches rest");
            }
            speed=6; distance=StationLoop.AdvanceToStop(ref speed,100,0,3,3);
            Near((float)distance,6,"Manual service pause brakes smoothly before the next stop"); Near(speed,0,"Paused service is stationary");
            speed=0; distance=StationLoop.AdvanceToStop(ref speed,0,12,3,1); Near((float)distance,0,"Docked train stays at the platform");
            Reject(()=>StationLoop.AdvanceToStop(ref speed,double.NaN,12,3,1),"Reject a nonfinite station distance.");
            Reject(()=>route.LocalPose(double.NaN),"Reject nonfinite route distance.");
        }
        finally {UnityEngine.Object.DestroyImmediate(go);}
    }
}
