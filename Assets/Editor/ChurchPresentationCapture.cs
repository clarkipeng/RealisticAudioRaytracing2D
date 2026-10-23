using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;

public static class ChurchPresentationCapture
{
    const string ScenePath = "Assets/Scenes/Church.unity";
    const int Width = 1280;
    const int Height = 720;
    const int Fps = 24;
    const float Duration = 8f;
    const string MaterialFolder = "Assets/Materials/ChurchPresentation";
    const string AvatarRoot = "Assets/External/Kenney/AnimatedCharactersProtagonists";
    const string ControllerPath = "Assets/External/Kenney/AnimatedCharactersProtagonists/ChurchAvatar.controller";

    [MenuItem("Tools/Acoustics/Install church presentation")]
    public static void Install()
    {
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        InstallIntoScene();
    }

    static void InstallIntoScene()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        var session = FindObject<SceneSession>();
        if (session == null) throw new InvalidOperationException("Church scene has no SceneSession.");
        Directory.CreateDirectory(MaterialFolder);
        var low = CreateLowPolyMaterials();
        var preset = FindObject<ChurchVisualPreset>();
        if (preset == null) preset = session.gameObject.AddComponent<ChurchVisualPreset>();
        if (preset.bindings.Length > 0) preset.Apply(ChurchVisualPreset.Preset.Realistic);
        InstallPostProcess(session);
        InstallAvatar(session);
        var bindings = new List<ChurchVisualPreset.Binding>();
        foreach (var root in scene.GetRootGameObjects())
        foreach (var renderer in root.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer.GetComponentInParent<ChurchAvatarPresenter>() != null) continue;
            var original = renderer.sharedMaterials;
            var stylized = new Material[original.Length];
            for (int i = 0; i < original.Length; i++) stylized[i] = low[MaterialKind(renderer, original[i])];
            bindings.Add(new ChurchVisualPreset.Binding {renderer = renderer, realistic = original, lowPoly = stylized});
        }
        preset.bindings = bindings.ToArray();
        preset.Apply(ChurchVisualPreset.Preset.Realistic);
        EditorUtility.SetDirty(session.gameObject);
        EditorUtility.SetDirty(session.player.gameObject);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        AssetDatabase.SaveAssets();
        Debug.Log("Installed Church visual presets and movement presentation animator.");
    }

    public static void CaptureBoth()
    {
        if (!Application.isBatchMode) throw new InvalidOperationException("Church presentation capture is batch-only.");
        InstallIntoScene();
        string output = Argument("-scenePresentationOutput");
        if (string.IsNullOrEmpty(output)) output = Path.GetFullPath(".context/scene-presentation/footage");
        Directory.CreateDirectory(output);
        output = FreshRunDirectory(output);

        var summaries = new List<string>();
        foreach (ChurchVisualPreset.Preset preset in Enum.GetValues(typeof(ChurchVisualPreset.Preset)))
            summaries.Add(CapturePreset(preset, output));
        bool smoke = Argument("-scenePresentationSmoke") == "1";
        var routeDelta = smoke ? 0f : CompareRoutes(output);
        File.WriteAllText(Path.Combine(output, "summary.json"), "{\n\"maxRouteDeltaMetres\":" + routeDelta.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + ",\n\"captures\":[\n" + string.Join(",\n", summaries) + "\n]\n}\n");
        Debug.Log("Church presentation capture complete: " + output);
    }

    static string CapturePreset(ChurchVisualPreset.Preset preset, string root)
    {
        string presetName = preset.ToString();
        string directory = Path.Combine(root, presetName);
        if (Directory.Exists(directory)) throw new IOException("Capture output already exists: " + directory);
        Directory.CreateDirectory(directory);
        string firstFrames = Path.Combine(directory, "first-person-frames");
        string thirdFrames = Path.Combine(directory, "third-person-frames");
        Directory.CreateDirectory(firstFrames);
        Directory.CreateDirectory(thirdFrames);
        var oldSimulation = Physics.simulationMode;
        float oldFixed = Time.fixedDeltaTime;
        int oldVsync = QualitySettings.vSyncCount;
        int oldAa = QualitySettings.antiAliasing;
        ShadowQuality oldShadows = QualitySettings.shadows;
        ShadowResolution oldShadowResolution = QualitySettings.shadowResolution;
        float oldShadowDistance = QualitySettings.shadowDistance;
        var samples = new List<Vector3>();
        string colliderHashBefore = null;
        string colliderHashAfter = null;
        string acousticHashBefore = null;
        string acousticHashAfter = null;
        float minLuma = 1f, maxLuma = 0f;
        bool smoke = Argument("-scenePresentationSmoke") == "1";
        int frameCount = smoke ? 4 : Mathf.RoundToInt(Duration * Fps);
        Vector3 final = Vector3.zero;
        bool grounded = false;
        try
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            Physics.simulationMode = SimulationMode.Script;
            Time.fixedDeltaTime = .02f;
            QualitySettings.vSyncCount = 0;
            QualitySettings.antiAliasing = 4;
            QualitySettings.shadows = ShadowQuality.All;
            QualitySettings.shadowResolution = ShadowResolution.High;
            QualitySettings.shadowDistance = 90f;

            var session = FindObject<SceneSession>();
            if (session == null) throw new InvalidOperationException("Church scene has no SceneSession.");
            var visual = FindObject<ChurchVisualPreset>();
            if (visual == null) visual = session.gameObject.AddComponent<ChurchVisualPreset>();
            var avatar = session.player.GetComponent<ChurchAvatarPresenter>();
            foreach (ChurchVisualPreset.Preset style in Enum.GetValues(typeof(ChurchVisualPreset.Preset)))
            {
                visual.Apply(style);
                foreach (var binding in visual.bindings)
                {
                    var expected = style == ChurchVisualPreset.Preset.Realistic ? binding.realistic : binding.lowPoly;
                    var actual = binding.renderer.sharedMaterials;
                    if (actual.Length != expected.Length) throw new InvalidOperationException("Material slots changed after reload.");
                    for (int i = 0; i < expected.Length; i++)
                        if (expected[i] == null || actual[i] != expected[i]) throw new InvalidOperationException("Preset material missing after reload.");
                }
            }
            visual.Apply(preset);
            session.InitializeSimulation();
            avatar.Initialize();
            avatar.animator.Rebind();
            avatar.animator.Update(0);
            colliderHashBefore = ColliderHash();
            acousticHashBefore = AcousticHash();

            var firstPerson = session.player.GetComponentInChildren<Camera>(true);
            if (firstPerson == null) throw new InvalidOperationException("Player has no gameplay camera.");
            var thirdPerson = CreateThirdPersonCamera();
            PrepareCaptureCamera(firstPerson);
            var rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
            var texture = new Texture2D(Width, Height, TextureFormat.RGB24, false);
            firstPerson.targetTexture = rt;
            thirdPerson.targetTexture = rt;

            float fixedTime = 0f;
            using (var inputLog = new StreamWriter(Path.Combine(directory, "route.csv"), false, Encoding.UTF8))
            {
                inputLog.WriteLine("tick,time,x,y,z,grounded,inputX,inputY,lookX,lookY,interact,animationTime,bonePose");
                for (int frame = 0; frame < frameCount; frame++)
                {
                    float frameTime = frame / (float)Fps;
                    while (fixedTime + Time.fixedDeltaTime <= frameTime + .0001f)
                    {
                        var input = RouteInput(fixedTime);
                        session.Step(input, Time.fixedDeltaTime);
                        Physics.Simulate(Time.fixedDeltaTime);
                        avatar.ManualUpdate(Time.fixedDeltaTime);
                        samples.Add(session.player.position);
                        inputLog.WriteLine(string.Join(",", session.Tick, fixedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            session.player.position.x.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            session.player.position.y.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            session.player.position.z.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            session.Walker.Ground != null ? "1" : "0",
                            input.movement.x.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            input.movement.y.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            input.look.x.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            input.look.y.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            input.interact ? "1" : "0",
                            avatar.animator.GetCurrentAnimatorStateInfo(0).normalizedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                            BonePose(avatar)));
                        fixedTime += Time.fixedDeltaTime;
                    }
                    avatar.SetVisible(false);
                    CaptureFrame(firstPerson, rt, texture, Path.Combine(firstFrames, $"frame_{frame:0000}.png"), ref minLuma, ref maxLuma);
                    avatar.SetVisible(true);
                    PositionCamera(thirdPerson, session.player, frameTime);
                    CaptureFrame(thirdPerson, rt, texture, Path.Combine(thirdFrames, $"frame_{frame:0000}.png"), ref minLuma, ref maxLuma);
                    avatar.SetVisible(false);
                }
            }
            firstPerson.targetTexture = null;
            thirdPerson.targetTexture = null;
            RenderTexture.active = null;
            UnityEngine.Object.DestroyImmediate(texture);
            UnityEngine.Object.DestroyImmediate(rt);
            UnityEngine.Object.DestroyImmediate(thirdPerson.gameObject);
            final = session.player.position;
            grounded = session.Walker.Ground != null || Physics.Raycast(final + Vector3.up * .2f, Vector3.down, 1.0f);
            colliderHashAfter = ColliderHash();
            acousticHashAfter = AcousticHash();
        }
        finally
        {
            Physics.simulationMode = oldSimulation;
            Time.fixedDeltaTime = oldFixed;
            QualitySettings.vSyncCount = oldVsync;
            QualitySettings.antiAliasing = oldAa;
            QualitySettings.shadows = oldShadows;
            QualitySettings.shadowResolution = oldShadowResolution;
            QualitySettings.shadowDistance = oldShadowDistance;
        }
        float distance = 0f;
        for (int i = 1; i < samples.Count; i++) distance += Vector3.Distance(samples[i - 1], samples[i]);
        if (distance < (smoke ? .05f : 8f)) throw new InvalidOperationException("Capture route did not move far enough.");
        if (!smoke && !grounded) throw new InvalidOperationException("Capture route ended off the ground.");
        if (colliderHashBefore != colliderHashAfter || acousticHashBefore != acousticHashAfter) throw new InvalidOperationException("Capture changed collider or acoustic scene hashes.");
        string movie = Path.Combine(directory, presetName + "-first-person-gameplay.mp4");
        string thirdMovie = Path.Combine(directory, presetName + "-third-person-animation.mp4");
        string ffmpeg = FindExecutable("ffmpeg");
        bool encoded = ffmpeg != null && Encode(ffmpeg, firstFrames, movie);
        bool thirdEncoded = ffmpeg != null && Encode(ffmpeg, thirdFrames, thirdMovie);
        File.Copy(Path.Combine(firstFrames, $"frame_{frameCount / 2:0000}.png"), Path.Combine(root, presetName + "-first-person-mid.png"), true);
        File.Copy(Path.Combine(firstFrames, $"frame_{frameCount - 1:0000}.png"), Path.Combine(root, presetName + "-first-person-final.png"), true);
        File.Copy(Path.Combine(thirdFrames, $"frame_{frameCount / 2:0000}.png"), Path.Combine(root, presetName + "-third-person-mid.png"), true);
        return JsonUtility.ToJson(new CaptureSummary
        {
            preset = presetName,
            frameCount = frameCount,
            fps = Fps,
            width = Width,
            height = Height,
            seconds = frameCount / (float)Fps,
            finalPosition = final,
            minLuma = minLuma,
            maxLuma = maxLuma,
            travelledMetres = distance,
            colliderHash = colliderHashAfter,
            acousticHash = acousticHashAfter,
            framesDirectory = firstFrames,
            thirdPersonFramesDirectory = thirdFrames,
            moviePath = encoded ? movie : "",
            thirdPersonMoviePath = thirdEncoded ? thirdMovie : "",
            encodedMovie = encoded,
            encodedThirdPersonMovie = thirdEncoded
        }, true);
    }

    static void PrepareCaptureCamera(Camera camera)
    {
        camera.clearFlags = CameraClearFlags.Skybox;
        camera.fieldOfView = 70f;
        camera.nearClipPlane = .05f;
        camera.farClipPlane = 180f;
        camera.allowHDR = true;
        var grade = camera.GetComponent<ChurchPresentationGrade>();
        if (grade == null) grade = camera.gameObject.AddComponent<ChurchPresentationGrade>();
        grade.shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/Shaders/ChurchPresentationGrade.shader");
    }

    static Camera CreateThirdPersonCamera()
    {
        var go = new GameObject("Presentation gameplay capture camera");
        var camera = go.AddComponent<Camera>();
        camera.enabled = false;
        camera.clearFlags = CameraClearFlags.Skybox;
        camera.fieldOfView = 58f;
        camera.nearClipPlane = .05f;
        camera.farClipPlane = 190f;
        camera.allowHDR = true;
        var grade = go.AddComponent<ChurchPresentationGrade>();
        grade.shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/Shaders/ChurchPresentationGrade.shader");
        return camera;
    }

    static void PositionCamera(Camera camera, Transform player, float time)
    {
        Vector3 pivot = player.position + Vector3.up * 1.2f + player.forward * Mathf.Lerp(1.8f, 3.6f, Mathf.Clamp01((time - 1f) / 5f));
        Vector3 offset = -player.forward * 5.2f + Vector3.up * 2.0f + player.right * Mathf.Sin(time * .45f) * .65f;
        camera.transform.position = Vector3.Lerp(camera.transform.position == Vector3.zero ? player.position + offset : camera.transform.position, player.position + offset, .85f);
        camera.transform.rotation = Quaternion.LookRotation(pivot - camera.transform.position, Vector3.up);
    }

    static SceneRecording.Input RouteInput(float time)
    {
        var input = new SceneRecording.Input();
        input.interact = false;
        if (time < 4.8f) input.movement = new Vector2(0f, 1f);
        else if (time < 5.9f) input.movement = new Vector2(.45f, .75f);
        else input.movement = new Vector2(-.18f, .92f);
        float yaw = time < 2.4f ? 0f : time < 5.4f ? .11f : -.16f;
        float pitch = time < 3.0f ? -.035f : time < 6.0f ? .018f : -.010f;
        input.look = new Vector2(yaw, pitch);
        return input;
    }

    static void SampleLuma(Texture2D texture, ref float min, ref float max)
    {
        var pixels = texture.GetPixels32();
        for (int i = 0; i < pixels.Length; i += 997)
        {
            var p = pixels[i];
            float luma = (p.r * .2126f + p.g * .7152f + p.b * .0722f) / 255f;
            min = Mathf.Min(min, luma);
            max = Mathf.Max(max, luma);
        }
    }

    static void CaptureFrame(Camera camera, RenderTexture rt, Texture2D texture, string path, ref float minLuma, ref float maxLuma)
    {
        camera.Render();
        RenderTexture.active = rt;
        texture.ReadPixels(new Rect(0, 0, Width, Height), 0, 0);
        texture.Apply(false);
        SampleLuma(texture, ref minLuma, ref maxLuma);
        File.WriteAllBytes(path, texture.EncodeToPNG());
    }

    static Dictionary<string, Material> CreateLowPolyMaterials()
    {
        var result = new Dictionary<string, Material>();
        Add("stone", new Color(.55f, .58f, .60f), 0f, .42f);
        Add("wood", new Color(.56f, .32f, .16f), 0f, .32f);
        Add("glass", new Color(.34f, .72f, .95f, .82f), 0f, .72f);
        Add("flame", new Color(1f, .54f, .17f), 0f, .15f, new Color(1.25f, .55f, .12f));
        Add("wax", new Color(.92f, .84f, .64f), 0f, .28f);
        Add("brass", new Color(.92f, .66f, .25f), .25f, .62f);
        Add("iron", new Color(.30f, .34f, .39f), .35f, .48f);
        Add("floor", new Color(.42f, .43f, .40f), 0f, .35f);
        Add("earth", new Color(.22f, .35f, .22f), 0f, .22f);
        return result;

        void Add(string key, Color color, float metallic, float gloss, Color? emission = null)
        {
            string path = $"{MaterialFolder}/LowPoly-{key}.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Standard")) { name = "LowPoly-" + key };
                AssetDatabase.CreateAsset(material, path);
            }
            material.color = color;
            material.SetFloat("_Metallic", metallic);
            material.SetFloat("_Glossiness", gloss);
            if (emission.HasValue)
            {
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", emission.Value);
            }
            else
            {
                material.DisableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", Color.black);
            }
            EditorUtility.SetDirty(material);
            result[key] = material;
        }
    }

    static void InstallPostProcess(SceneSession session)
    {
        var camera = session.player.GetComponentInChildren<Camera>(true);
        if (camera == null) return;
        var grade = camera.GetComponent<ChurchPresentationGrade>();
        if (grade == null) grade = camera.gameObject.AddComponent<ChurchPresentationGrade>();
        grade.shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/Shaders/ChurchPresentationGrade.shader");
        grade.exposure = .25f;
        grade.vignette = .18f;
        EditorUtility.SetDirty(camera.gameObject);
    }

    static void InstallAvatar(SceneSession session)
    {
        ConfigureClip(AvatarRoot + "/Animations/idle.fbx");
        ConfigureClip(AvatarRoot + "/Animations/run.fbx");
        string modelPath = AvatarRoot + "/Model/characterMedium.fbx";
        string texturePath = AvatarRoot + "/Skins/skaterMaleA.png";
        var importer = (ModelImporter)AssetImporter.GetAtPath(modelPath);
        if (importer.avatarSetup != ModelImporterAvatarSetup.CreateFromThisModel)
        {
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.SaveAndReimport();
        }
        var model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
        if (model == null) throw new FileNotFoundException("Missing Kenney avatar model: " + modelPath);
        var controller = BuildAvatarController();
        var avatarRoot = session.player.Find("Presentation avatar");
        if (avatarRoot == null)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(model, session.player);
            avatarRoot = instance.transform;
            avatarRoot.name = "Presentation avatar";
        }
        avatarRoot.localPosition = Vector3.zero;
        avatarRoot.localRotation = Quaternion.identity;
        avatarRoot.localScale = Vector3.one;
        var body = avatarRoot.GetComponentInChildren<SkinnedMeshRenderer>();
        var capsule = session.player.GetComponent<CharacterController>();
        // Fit the mesh bounds, not the padded animated culling bounds.
        var local = body.sharedMesh.bounds;
        var bounds = new Bounds(body.transform.TransformPoint(local.center), Vector3.zero);
        for (int i = 0; i < 8; i++)
            bounds.Encapsulate(body.transform.TransformPoint(local.center + Vector3.Scale(local.extents,
                new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1))));
        float scale = capsule.height * session.player.lossyScale.y / bounds.size.y;
        avatarRoot.localScale = Vector3.one * scale;
        avatarRoot.localPosition = Vector3.up * (capsule.center.y - capsule.height * .5f - capsule.skinWidth -
            (bounds.min.y - session.player.position.y) * scale / session.player.lossyScale.y);
        var animator = avatarRoot.GetComponent<Animator>();
        if (animator == null) animator = avatarRoot.gameObject.AddComponent<Animator>();
        foreach (var asset in AssetDatabase.LoadAllAssetsAtPath(modelPath))
            if (asset is Avatar avatar) animator.avatar = avatar;
        animator.runtimeAnimatorController = controller;
        animator.applyRootMotion = false;
        animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
        var material = AssetDatabase.LoadAssetAtPath<Material>(MaterialFolder + "/Kenney-skater.mat");
        if (material == null)
        {
            material = new Material(Shader.Find("Standard")) { name = "Kenney-skater" };
            AssetDatabase.CreateAsset(material, MaterialFolder + "/Kenney-skater.mat");
        }
        material.mainTexture = texture;
        material.SetFloat("_Glossiness", .36f);
        foreach (var renderer in avatarRoot.GetComponentsInChildren<Renderer>(true))
        {
            renderer.sharedMaterial = material;
            renderer.enabled = true;
        }
        var presenter = session.player.GetComponent<ChurchAvatarPresenter>();
        if (presenter == null) presenter = session.player.gameObject.AddComponent<ChurchAvatarPresenter>();
        presenter.animator = animator;
        presenter.renderers = avatarRoot.GetComponentsInChildren<Renderer>(true);
        presenter.SetVisible(false);
        EditorUtility.SetDirty(material);
        EditorUtility.SetDirty(avatarRoot.gameObject);
        EditorUtility.SetDirty(session.player.gameObject);
    }

    static void ConfigureClip(string path)
    {
        var importer = (ModelImporter)AssetImporter.GetAtPath(path);
        var clips = importer.clipAnimations.Length > 0 ? importer.clipAnimations : importer.defaultClipAnimations;
        bool changed = !importer.preserveHierarchy;
        importer.preserveHierarchy = true;
        foreach (var clip in clips) { changed |= !clip.loopTime; clip.loopTime = true; }
        if (changed) { importer.clipAnimations = clips; importer.SaveAndReimport(); }
    }

    static string MaterialKind(Renderer renderer, Material material)
    {
        string name = (renderer.name + " " + (material == null ? "" : material.name)).ToLowerInvariant();
        if (name.Contains("glass") || name.Contains("window")) return "glass";
        if (name.Contains("flame")) return "flame";
        if (name.Contains("wax")) return "wax";
        if (name.Contains("brass")) return "brass";
        if (name.Contains("iron") || name.Contains("metal")) return "iron";
        if (name.Contains("oak") || name.Contains("wood") || name.Contains("pew") || name.Contains("organ") || name.Contains("door")) return "wood";
        if (name.Contains("floor") || name.Contains("paving")) return "floor";
        if (name.Contains("meadow") || name.Contains("earth") || name.Contains("ground")) return "earth";
        return "stone";
    }

    static RuntimeAnimatorController BuildAvatarController()
    {
        var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(ControllerPath);
        if (controller == null) controller = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
        if (controller.layers.Length == 0) controller.AddLayer("Base Layer");
        if (controller.parameters.Length == 0)
        {
            controller.AddParameter("Speed", AnimatorControllerParameterType.Float);
            var idle = LoadClip(AvatarRoot + "/Animations/idle.fbx");
            var run = LoadClip(AvatarRoot + "/Animations/run.fbx");
            var root = controller.layers[0].stateMachine;
            var idleState = root.AddState("Idle");
            idleState.motion = idle;
            var runState = root.AddState("Run");
            runState.motion = run;
            root.defaultState = idleState;
            var toRun = idleState.AddTransition(runState);
            toRun.hasExitTime = false;
            toRun.duration = .12f;
            toRun.AddCondition(AnimatorConditionMode.Greater, .35f, "Speed");
            var toIdle = runState.AddTransition(idleState);
            toIdle.hasExitTime = false;
            toIdle.duration = .18f;
            toIdle.AddCondition(AnimatorConditionMode.Less, .25f, "Speed");
        }
        var layers = controller.layers;
        layers[0].defaultWeight = 1;
        controller.layers = layers;
        foreach (var child in controller.layers[0].stateMachine.states)
            child.state.motion = LoadClip(AvatarRoot + "/Animations/" + child.state.name.ToLowerInvariant() + ".fbx");
        foreach (var asset in AssetDatabase.LoadAllAssetsAtPath(ControllerPath)) EditorUtility.SetDirty(asset);
        AssetDatabase.SaveAssets();
        return controller;
    }

    static AnimationClip LoadClip(string path)
    {
        foreach (var asset in AssetDatabase.LoadAllAssetsAtPath(path))
            if (asset is AnimationClip clip && clip.name.Equals("Root|" + Path.GetFileNameWithoutExtension(path), StringComparison.OrdinalIgnoreCase)) return clip;
        throw new FileNotFoundException("Missing animation clip: " + path);
    }

    static string BonePose(ChurchAvatarPresenter avatar)
    {
        var text = new StringBuilder();
        foreach (var bone in avatar.GetComponentInChildren<SkinnedMeshRenderer>(true).bones)
            text.Append(bone.localRotation.ToString("R"));
        using var sha = SHA256.Create();
        return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(text.ToString())));
    }

    static string ColliderHash() => HashComponents("collider", root =>
    {
        var lines = new List<string>();
        foreach (var collider in root.GetComponentsInChildren<Collider>(true))
        {
            var line = collider.GetType().Name + "|" + TransformPath(collider.transform) + "|" + collider.enabled + "|" + collider.isTrigger;
            if (collider is BoxCollider box) line += "|box|" + box.center.ToString("R") + "|" + box.size.ToString("R");
            else if (collider is SphereCollider sphere) line += "|sphere|" + sphere.center.ToString("R") + "|" + sphere.radius.ToString("R");
            else if (collider is CapsuleCollider capsule) line += "|capsule|" + capsule.center.ToString("R") + "|" + capsule.radius.ToString("R") + "|" + capsule.height.ToString("R") + "|" + capsule.direction;
            else if (collider is CharacterController character) line += "|character|" + character.center.ToString("R") + "|" + character.radius.ToString("R") + "|" + character.height.ToString("R") + "|" + character.slopeLimit.ToString("R") + "|" + character.stepOffset.ToString("R");
            else if (collider is MeshCollider mesh) line += "|mesh|" + (mesh.sharedMesh == null ? "" : mesh.sharedMesh.name) + "|" + mesh.convex;
            lines.Add(line);
        }
        return lines;
    });

    static string AcousticHash() => HashComponents("acoustic", root =>
    {
        var lines = new List<string>();
        foreach (var surface in root.GetComponentsInChildren<AcousticSurface>(true))
            lines.Add(TransformPath(surface.transform) + "|" + surface.enabled + "|" + (surface.material == null ? "" : surface.material.name));
        return lines;
    });

    static string HashComponents(string label, Func<GameObject, List<string>> collect)
    {
        var lines = new List<string> { label };
        foreach (var root in SceneManager.GetActiveScene().GetRootGameObjects()) lines.AddRange(collect(root));
        lines.Sort(StringComparer.Ordinal);
        using var sha = SHA256.Create();
        return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(string.Join("\n", lines))));
    }

    static string TransformPath(Transform transform)
    {
        var names = new Stack<string>();
        while (transform != null)
        {
            names.Push(transform.name);
            transform = transform.parent;
        }
        return string.Join("/", names);
    }

    static T FindObject<T>() where T : UnityEngine.Object
    {
        foreach (var root in SceneManager.GetActiveScene().GetRootGameObjects())
        {
            var value = root.GetComponentInChildren<T>(true);
            if (value != null) return value;
        }
        return null;
    }

    static string Argument(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == name) return args[i + 1];
        return null;
    }

    static string FindExecutable(string name)
    {
        foreach (var folder in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            if (string.IsNullOrWhiteSpace(folder)) continue;
            string candidate = Path.Combine(folder, name);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    static bool Encode(string executable, string frames, string movie)
    {
        var arguments = $"-nostdin -y -threads 1 -framerate {Fps} -i \"{Path.Combine(frames, "frame_%04d.png")}\" -c:v libx264 -threads 1 -pix_fmt yuv420p -crf 18 \"{movie}\"";
        var info = new ProcessStartInfo(executable, arguments) { UseShellExecute = false, RedirectStandardError = true, RedirectStandardOutput = true };
        using (var process = Process.Start(info))
        {
            string stdout = null, stderr = null;
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            stdout = stdoutTask.GetAwaiter().GetResult();
            stderr = stderrTask.GetAwaiter().GetResult();
            if (process.ExitCode == 0) return true;
            Debug.LogWarning(stdout + "\n" + stderr);
            return false;
        }
    }

    static string FreshRunDirectory(string output)
    {
        for (int i = 0; i < 100; i++)
        {
            string path = Path.Combine(output, "run-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss") + (i == 0 ? "" : "-" + i));
            if (Directory.Exists(path)) continue;
            Directory.CreateDirectory(path);
            return path;
        }
        throw new IOException("Unable to allocate fresh output directory under " + output);
    }

    static float CompareRoutes(string output)
    {
        var paths = new[]
        {
            Path.Combine(output, ChurchVisualPreset.Preset.PolishedLowPoly.ToString(), "route.csv"),
            Path.Combine(output, ChurchVisualPreset.Preset.Realistic.ToString(), "route.csv")
        };
        if (!File.Exists(paths[0]) || !File.Exists(paths[1])) return -1f;
        var a = File.ReadAllLines(paths[0]);
        var b = File.ReadAllLines(paths[1]);
        if (a.Length != b.Length) throw new InvalidOperationException("Preset route logs have different lengths.");
        float max = 0f;
        for (int i = 1; i < a.Length; i++)
        {
            var av = ParsePosition(a[i]);
            var bv = ParsePosition(b[i]);
            max = Mathf.Max(max, Vector3.Distance(av, bv));
        }
        if (max > .001f) throw new InvalidOperationException("Preset routes diverged by " + max + " metres.");
        return max;
    }

    static Vector3 ParsePosition(string line)
    {
        var parts = line.Split(',');
        return new Vector3(float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture),
            float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture),
            float.Parse(parts[4], System.Globalization.CultureInfo.InvariantCulture));
    }

    [Serializable]
    sealed class CaptureSummary
    {
        public string preset;
        public int frameCount, fps, width, height;
        public float seconds, minLuma, maxLuma, travelledMetres;
        public Vector3 finalPosition;
        public string framesDirectory, thirdPersonFramesDirectory, moviePath, thirdPersonMoviePath, colliderHash, acousticHash;
        public bool encodedMovie, encodedThirdPersonMovie;
    }
}
