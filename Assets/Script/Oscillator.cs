using UnityEngine;

public class Oscillator : MonoBehaviour
{
    public float distance = 5f;
    public float speed = 2f;
    public Vector3 direction = Vector3.right;
    
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
        direction = direction.normalized;
    }

    void Update()
    {
        float offset = Mathf.Sin(Time.time * speed) * distance;
        transform.position = startPosition + direction * offset;
    }
}
