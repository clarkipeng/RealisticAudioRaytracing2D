using UnityEngine;

[RequireComponent(typeof(CharacterController))]
public class PlayerController : MonoBehaviour
{
    public float speed = 6f;
    public float gravity = -9.81f;
    public float jumpHeight = 2f;
    public bool requireLockedCursor;
    [System.NonSerialized] public bool externalClock;

    CharacterController controller;
    float verticalSpeed;
    public Transform Ground { get; private set; }
    Vector3 groundLocalPoint, previousGroundPoint, airborneVelocity;
    Quaternion previousGroundRotation;
    public void Simulate(Vector2 input,bool jump,float seconds) => Move(input,jump,seconds);

    void Awake()
    {
        controller = GetComponent<CharacterController>();
        controller.minMoveDistance = 0;
    }

    void Update()
    {
        if (externalClock || !controller.enabled) return;
        bool acceptInput = !requireLockedCursor || Cursor.lockState == CursorLockMode.Locked;
        Move(acceptInput ? new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical")) : Vector2.zero,
            acceptInput && Input.GetButtonDown("Jump"), Time.deltaTime);
    }

    void Move(Vector2 input, bool jump, float deltaTime)
    {
        if(deltaTime<=0) return;
        bool grounded=Ground!=null && controller.isGrounded;
        Vector3 groundVelocity=Vector3.zero;
        if(Ground!=null)
        {
            var carriedPosition=Ground.TransformPoint(groundLocalPoint);
            groundVelocity=(carriedPosition-previousGroundPoint)/deltaTime;
            var turn=Ground.rotation*Quaternion.Inverse(previousGroundRotation);
            // Carry in the support's frame before sweeping walking/gravity against its current colliders.
            transform.SetPositionAndRotation(carriedPosition,Quaternion.Euler(0,turn.eulerAngles.y,0)*transform.rotation);
            Physics.SyncTransforms();
        }
        if(grounded)
        {
            airborneVelocity=Vector3.zero;
            if(verticalSpeed<0) verticalSpeed=-2;
            if(jump) {verticalSpeed=Mathf.Sqrt(jumpHeight*-2*gravity); airborneVelocity=groundVelocity;}
        }
        verticalSpeed += gravity * deltaTime;
        input = Vector2.ClampMagnitude(input, 1);
        Vector3 motion = (transform.right * input.x + transform.forward * input.y) * speed + airborneVelocity;
        motion.y += verticalSpeed;
        Ground=null;
        var flags = controller.Move(motion * deltaTime);
        if ((flags & CollisionFlags.Above) != 0 && verticalSpeed > 0) verticalSpeed = 0;
        if(Ground!=null)
        {
            groundLocalPoint=Ground.InverseTransformPoint(transform.position);
            previousGroundPoint=transform.position; previousGroundRotation=Ground.rotation;
        }
        else if(grounded && !jump) airborneVelocity=groundVelocity;
    }

    void OnControllerColliderHit(ControllerColliderHit hit)
    {
        if(verticalSpeed<=0 && hit.normal.y>=Mathf.Cos(controller.slopeLimit*Mathf.Deg2Rad)) Ground=hit.collider.transform;
    }
}
