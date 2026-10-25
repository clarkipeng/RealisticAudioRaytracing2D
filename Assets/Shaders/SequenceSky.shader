Shader "Church/Sequence Sky" {
Properties { _Cloud("Cloud cover",Range(0,1))=0 _SequenceTime("Time",Float)=0 _Flash("Flash",Float)=0 _WindTravel("Wind travel",Vector)=(0,0,0,0) }
SubShader { Tags { "Queue"="Background" "RenderType"="Background" "PreviewType"="Skybox" } Cull Off ZWrite Off
Pass { CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
float _Cloud,_SequenceTime,_Flash;float4 _WindTravel;
struct v2f {float4 position:SV_POSITION;float3 direction:TEXCOORD0;};
v2f vert(float4 p:POSITION){v2f o;o.position=UnityObjectToClipPos(p);o.direction=p.xyz;return o;}
float hash(float2 p){return frac(sin(dot(p,float2(127.1,311.7)))*43758.5453);}
float noise(float2 p){float2 i=floor(p),f=frac(p);f=f*f*(3-2*f);return lerp(lerp(hash(i),hash(i+float2(1,0)),f.x),lerp(hash(i+float2(0,1)),hash(i+1),f.x),f.y);}
float4 frag(v2f i):SV_Target {
 float3 d=normalize(i.direction);float h=saturate(d.y);
 float3 sky=lerp(float3(.62,.77,.83),float3(.12,.38,.65),pow(h,.45));
 float2 p=d.xz/(max(.12,d.y)+.25)*2.8+_WindTravel.xz*.008;
 float n=noise(p)*.55+noise(p*2.1)*.27+noise(p*4.3)*.13+noise(p*8.7)*.05;
 float cover=smoothstep(lerp(.76,.15,_Cloud),lerp(.85,.35,_Cloud),n)*smoothstep(-.06,.12,d.y);
 sky=lerp(sky,lerp(float3(.88,.90,.88),float3(.20,.25,.30),_Cloud)*(.7+n*.5),cover);
 return float4(sky+_Flash*.25,1);
}
ENDCG } } Fallback Off }
