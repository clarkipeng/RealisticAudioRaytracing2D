Shader "Church/Sequence Foliage" {
Properties { _Color("Color",Color)=(.2,.4,.12,1) _Wind("Wind",Vector)=(1,0,0,0) _SequenceTime("Time",Float)=0 }
SubShader { Tags {"RenderType"="Opaque"} Cull Off
CGPROGRAM
#pragma surface surf Standard vertex:vert addshadow
#pragma target 3.0
fixed4 _Color;float4 _Wind;float _SequenceTime;
struct Input {float3 worldPos;};
void vert(inout appdata_full v) {
 float3 w=mul(unity_ObjectToWorld,v.vertex).xyz;
 float flexibility=saturate(v.vertex.y/7);float phase=dot(w.xz,float2(.6,.32));
 float3 shift=_Wind.xyz*(sin(_SequenceTime*1.7+phase)*.035+sin(_SequenceTime*5.2+phase*3)*.012)*flexibility;
 v.vertex.xyz+=mul((float3x3)unity_WorldToObject,shift);
}
void surf(Input i,inout SurfaceOutputStandard o){o.Albedo=_Color.rgb;o.Metallic=0;o.Smoothness=.18;}
ENDCG
} Fallback "Diffuse" }
