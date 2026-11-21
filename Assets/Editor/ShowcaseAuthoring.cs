using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Authors each live showcase scene's PathLiveScene: the sources that play and the smoke run's walk. Rerunnable.
//   Acoustics > Author live showcase scenes, or batch mode: -executeMethod ShowcaseAuthoring.Apply
public static class ShowcaseAuthoring
{
    [MenuItem("Acoustics/Author live showcase scenes")]
    public static void Apply()
    {
        Author("Assets/Scenes/Church.unity", Church);
        Author("Assets/Scenes/TrainStation.unity", Station);
        Debug.Log("SHOWCASE_AUTHORING done");
        if (Application.isBatchMode) EditorApplication.Exit(0);
    }

    static void Author(string path, Action<SceneSession, PathLiveScene, Vector3> configure)
    {
        var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
        var session = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<SceneSession>(true)).Single();
        var spawn = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).Single(t => t.name == "Player spawn 0");
        var showcase = session.GetComponent<PathLiveScene>();
        if (showcase == null) showcase = session.gameObject.AddComponent<PathLiveScene>();
        configure(session, showcase, spawn.position);
        EditorUtility.SetDirty(showcase);
        EditorSceneManager.SaveScene(scene);
    }

    // The organ, and the walk of README part 38: to the organ (stopping short of it), to the doors' centre from inside,
    // open them, out through them and along the facade.
    static void Church(SceneSession session, PathLiveScene showcase, Vector3 spawn)
    {
        var sources = session.manager.takeSources.Where(s => s != null);
        var organ = sources.OrderBy(s => s.name == "Pipe organ" ? 0 : 1).ThenBy(s => (s.transform.position - new Vector3(0, 5, 18)).sqrMagnitude).First();
        var doors = session.actors.OfType<SlidingDoor>().ToArray();
        var door = doors.Aggregate(Vector3.zero, (a, d) => a + d.transform.position) / doors.Length;
        Vector3 Flat(Vector3 v) => new Vector3(v.x, spawn.y, v.z);
        var inward = Flat(organ.transform.position) - Flat(door); inward.y = 0; inward.Normalize();
        var along = Vector3.Cross(Vector3.up, inward);
        showcase.sources = showcase.startLooped = new[] {organ};
        showcase.route = new[] {Flat(organ.transform.position) - inward * 3, Flat(door) + inward * 1.2f, Flat(door) - inward * 8, Flat(door) - inward * 8 + along * 8};
        showcase.interactAfter = 1; showcase.standSeconds = 5; showcase.holdSeconds = 2;
        showcase.population = 0;
    }

    // The train and the waiting room's signal. The walk: along the platform to the waiting room's door (within the door's
    // 2.2 m reach), open it, in
    // toward the signal, back out, then to the platform edge to hear the train leave and come round again (it waits
    // 10 s at the platform, then laps the loop in about 25 s).
    static void Station(SceneSession session, PathLiveScene showcase, Vector3 spawn)
    {
        showcase.sources = showcase.startLooped = session.startSounds.ToArray();
        float y = spawn.y;
        showcase.route = new[] {new Vector3(5.6f, y, 4.4f), new Vector3(5.6f, y, 11f), new Vector3(5.6f, y, 2.5f), new Vector3(8.8f, y, -.5f)};
        showcase.interactAfter = 0; showcase.standSeconds = 5; showcase.holdSeconds = 25;
        // Two sources: 4,000 rays each keeps the 8,000 in flight of the Church's one source.
        showcase.population = 4000;
    }
}
