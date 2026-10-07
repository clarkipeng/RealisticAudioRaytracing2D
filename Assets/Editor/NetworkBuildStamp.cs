using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

// The editor and standalone reject connections from different project inputs.
public sealed class NetworkBuildStamp : IPreprocessBuildWithReport
{
    public int callbackOrder=>0;
    public void OnPreprocessBuild(BuildReport report)
    {
        File.WriteAllText("Assets/Resources/NetworkBuildStamp.txt",MultiplayerSession.BuildFingerprint());
        AssetDatabase.ImportAsset("Assets/Resources/NetworkBuildStamp.txt",ImportAssetOptions.ForceSynchronousImport);
    }
}
