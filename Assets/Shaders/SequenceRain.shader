Shader "Church/Sequence Rain" {
SubShader {Tags {"Queue"="Transparent" "RenderType"="Transparent"} Blend SrcAlpha OneMinusSrcAlpha ZWrite Off Cull Off
Pass {CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
struct v2f{float4 position:SV_POSITION;fixed4 color:COLOR;UNITY_FOG_COORDS(0)};
v2f vert(float4 p:POSITION,fixed4 c:COLOR){v2f o;o.position=UnityObjectToClipPos(p);o.color=c;UNITY_TRANSFER_FOG(o,o.position);return o;}
fixed4 frag(v2f i):SV_Target {fixed4 c=i.color;UNITY_APPLY_FOG(i.fogCoord,c);return c;}
ENDCG } } }
