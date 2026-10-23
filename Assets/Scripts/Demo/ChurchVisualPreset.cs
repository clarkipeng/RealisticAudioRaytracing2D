using System;
using UnityEngine;
using UnityEngine.Rendering;

// Both material sets are scene assets, so switching survives save/reload and domain reload.
[DisallowMultipleComponent]
public sealed class ChurchVisualPreset : MonoBehaviour
{
    public enum Preset { PolishedLowPoly, Realistic }
    [Serializable]
    public struct Binding
    {
        public Renderer renderer;
        public Material[] realistic, lowPoly;
    }
    public Preset activePreset = Preset.Realistic;
    public Binding[] bindings = Array.Empty<Binding>();
    Preset applied;

    void Awake() => Apply(activePreset);
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F7)) activePreset = Preset.PolishedLowPoly;
        if (Input.GetKeyDown(KeyCode.F8)) activePreset = Preset.Realistic;
        if (activePreset != applied) Apply(activePreset);
    }

    public void Apply(Preset preset)
    {
        applied = activePreset = preset;
        foreach (var binding in bindings)
            if (binding.renderer != null)
                binding.renderer.sharedMaterials = preset == Preset.Realistic ? binding.realistic : binding.lowPoly;
        ConfigureLights(preset);
        ConfigureRenderSettings(preset);
    }

    void ConfigureLights(Preset preset)
    {
        foreach (var root in gameObject.scene.GetRootGameObjects())
        foreach (var light in root.GetComponentsInChildren<Light>(true))
        {
            string name = light.name.ToLowerInvariant();
            light.shadows = LightShadows.Soft;
            if (name.Contains("moonlight"))
            {
                light.intensity = preset == Preset.Realistic ? .13f : .34f;
                light.color = preset == Preset.Realistic ? new Color(.50f, .62f, .92f) : new Color(.62f, .82f, 1f);
                light.shadowResolution = LightShadowResolution.High;
            }
            else if (name.Contains("candle"))
            {
                light.range = preset == Preset.Realistic ? 13f : 10f;
                light.intensity = preset == Preset.Realistic ? 2.1f : 1.15f;
                light.color = preset == Preset.Realistic ? new Color(1f, .67f, .34f) : new Color(1f, .76f, .38f);
                light.renderMode = LightRenderMode.ForcePixel;
            }
            else if (name.Contains("window"))
            {
                light.range = preset == Preset.Realistic ? 28f : 24f;
                light.intensity = preset == Preset.Realistic ? 1.75f : 2.45f;
                light.color = preset == Preset.Realistic ? new Color(.40f, .56f, 1f) : new Color(.26f, .78f, 1f);
                light.shadowResolution = LightShadowResolution.High;
                light.renderMode = LightRenderMode.ForcePixel;
            }
        }
    }

    static void ConfigureRenderSettings(Preset preset)
    {
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Linear;
        if (preset == Preset.Realistic)
        {
            RenderSettings.ambientSkyColor = new Color(.10f, .12f, .18f);
            RenderSettings.ambientEquatorColor = new Color(.045f, .055f, .075f);
            RenderSettings.ambientGroundColor = new Color(.018f, .020f, .024f);
            RenderSettings.fogStartDistance = 60f;
            RenderSettings.fogEndDistance = 180f;
            RenderSettings.fogColor = new Color(.040f, .055f, .080f);
        }
        else
        {
            RenderSettings.ambientSkyColor = new Color(.22f, .29f, .35f);
            RenderSettings.ambientEquatorColor = new Color(.11f, .15f, .17f);
            RenderSettings.ambientGroundColor = new Color(.050f, .060f, .055f);
            RenderSettings.fogStartDistance = 80f;
            RenderSettings.fogEndDistance = 210f;
            RenderSettings.fogColor = new Color(.075f, .12f, .13f);
        }
        // Explicit ambient probe also works during synchronous offscreen rendering.
        var ambient = new SphericalHarmonicsL2();
        ambient.AddAmbientLight(RenderSettings.ambientSkyColor * .6f + RenderSettings.ambientEquatorColor * .4f);
        RenderSettings.ambientProbe = ambient;
    }
}
