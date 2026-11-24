using UnityEngine;

public class Oscillator : MonoBehaviour
{
    public float distance = 5f;
    public bool limitDistance = true;
    public float speed = 5f;
    public Vector3 direction = Vector3.right;
    public bool moving = false;
    public KeyCode toggleKey = KeyCode.M;
    public RaytracedAudioSource raytracedAudioSource;
    public RayTraceManager rayTraceManager;
    public bool startAudioOnMove = true;
    public bool startAudioOnlyOnce = true;
    
    Vector3 startPosition;
    bool startedAudio;

    void Start()
    {
        startPosition = transform.position;
        if (direction.sqrMagnitude < 1e-6f) direction = Vector3.right;
        direction = direction.normalized;
        if (raytracedAudioSource == null)
            raytracedAudioSource = GetComponent<RaytracedAudioSource>();
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
            ToggleMoving();

        if (moving)
        {
            if (!limitDistance)
            {
                transform.position += direction * (speed * Time.deltaTime);
                return;
            }

            float traveled = Vector3.Dot(transform.position - startPosition, direction);
            float remaining = distance - traveled;
            if (remaining <= 0f)
            {
                transform.position = startPosition + direction * distance;
                moving = false;
                return;
            }

            float offset = Mathf.Min(speed * Time.deltaTime, remaining);
            transform.position += direction * offset;
        }
    }

    public void ToggleMoving()
    {
        if (!moving)
        {
            if (limitDistance)
            {
                float traveled = Vector3.Dot(transform.position - startPosition, direction);
                if (traveled >= distance - 0.001f)
                    transform.position = startPosition;
            }

            StartAudioIfNeeded();
        }

        moving = !moving;
    }

    void StartAudioIfNeeded()
    {
        if (!startAudioOnMove) return;
        if (startAudioOnlyOnce && startedAudio) return;

        if (raytracedAudioSource != null)
            raytracedAudioSource.Play();
        else if (rayTraceManager != null)
            rayTraceManager.StartStreaming();

        startedAudio = true;
    }
}
