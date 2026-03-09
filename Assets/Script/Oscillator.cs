using UnityEngine;

public class Oscillator : MonoBehaviour
{
    public float distance = 5f;
    public float speed = 200f;
    public Vector3 direction = Vector3.right;
    public bool moving = false;
    
    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
        direction = direction.normalized;
    }

    void Update()
    {
        if (moving)
        {
            float offset = speed * Time.deltaTime;
            transform.position += direction * offset;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            moving = !moving;
        }


        Debug.Log($"Current position: {transform.position}, start position: {startPosition}, distance: {startPosition.x + distance}.");
        if (transform.position.x > startPosition.x + distance)
        {
            transform.position = startPosition;
            moving = false;
        }
    }
}
