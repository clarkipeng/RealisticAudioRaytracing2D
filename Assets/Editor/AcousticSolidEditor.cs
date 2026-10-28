using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(AcousticSolid))]
public sealed class AcousticSolidEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var status = AcousticSolidBuilder.Describe((AcousticSolid)target);
        var type = status.state == AcousticSolidBuilder.State.Failed ? MessageType.Error :
            status.state == AcousticSolidBuilder.State.Inactive ? MessageType.Warning : MessageType.Info;
        EditorGUILayout.HelpBox(status.message, type);
        if (status.state == AcousticSolidBuilder.State.Failed && GUILayout.Button("Retry")) AcousticSolidBuilder.Retry();
        EditorGUILayout.HelpBox("Select fixed child surfaces that form one interior medium. The exterior regenerates automatically when members, " +
            "their relative arrangement or materials change; moving the whole group does not. Unlisted surfaces, including moving doors, remain independent.", MessageType.None);
    }

    [SettingsProvider]
    static SettingsProvider Preferences() => new SettingsProvider("Preferences/Acoustics", SettingsScope.User)
    {
        label = "Acoustics",
        keywords = new[] { "solid", "boundary", "python", "manifold" },
        guiHandler = _ =>
        {
            string current = EditorPrefs.GetString(AcousticSolidBuilder.PythonPreference, "python3");
            string next = EditorGUILayout.DelayedTextField("Solid Boundary Python", current);
            if (next != current) { EditorPrefs.SetString(AcousticSolidBuilder.PythonPreference, next); AcousticSolidBuilder.Retry(); }
            if (!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable(AcousticSolidBuilder.PythonVariable)))
                EditorGUILayout.HelpBox(AcousticSolidBuilder.PythonVariable + " overrides this setting: " + AcousticSolidBuilder.Python, MessageType.Warning);
            EditorGUILayout.HelpBox("Python environment with tools/solid-boundary-requirements.txt. It regenerates AcousticSolid exteriors after edits. " +
                "Scenes whose saved exteriors are current capture without it.", MessageType.Info);
        }
    };
}
