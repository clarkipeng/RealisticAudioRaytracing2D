using UnityEngine;

public class MouseLook : MonoBehaviour
{
    [Range(30,360)] public float sensitivity = 180f;
    public Transform playerBody;
    public bool lockOnStart = true;
    [System.NonSerialized] public bool externalClock;

    float xRotation = 0f;
    public float Pitch => xRotation;
    public void Initialize() { xRotation=Mathf.DeltaAngle(0,transform.localEulerAngles.x); }
    public Vector2 ReadDelta() => new Vector2(Input.GetAxis("Mouse X"),Input.GetAxis("Mouse Y"))*sensitivity/60f;

    void Start()
    {
        if(lockOnStart) Cursor.lockState = CursorLockMode.Locked;
    }

    void Update()
    {
        if(externalClock || Cursor.lockState != CursorLockMode.Locked) return;
        // Mouse axes already contain frame displacement. Keep the old 60 Hz calibration,
        // without making a physical mouse movement slower when rendering gets faster.
        Simulate(ReadDelta());
    }
    public void Simulate(Vector2 degrees)
    {
        xRotation -= degrees.y;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);

        transform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
        playerBody.Rotate(Vector3.up * degrees.x);
    }
}
