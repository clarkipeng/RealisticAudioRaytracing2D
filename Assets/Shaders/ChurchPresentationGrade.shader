Shader "Hidden/ChurchPresentationGrade"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Contrast ("Contrast", Float) = 1
        _Exposure ("Exposure", Float) = 0
        _Saturation ("Saturation", Float) = 1
        _Vignette ("Vignette", Float) = 0
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float _Contrast, _Exposure, _Saturation, _Vignette;

            fixed4 frag(v2f_img i) : SV_Target
            {
                float3 color = tex2D(_MainTex, i.uv).rgb;
                color *= exp2(_Exposure);
                color = .18 * pow(max(color / .18, 0), _Contrast);
                float luma = dot(color, float3(.2126, .7152, .0722));
                color = lerp(luma.xxx, color, _Saturation);
                float2 centered = i.uv * 2 - 1;
                float mask = 1 - smoothstep(.25, 1.35, dot(centered, centered));
                color *= lerp(1, mask, _Vignette);
                return fixed4(saturate(color), 1);
            }
            ENDCG
        }
    }
}
