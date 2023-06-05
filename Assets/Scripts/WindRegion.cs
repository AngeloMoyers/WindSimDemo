using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WindRegion : MonoBehaviour
{
    [SerializeField] GameObject ArrowObject;
    [SerializeField] ParticleSystemForceField ForceField;

    [SerializeField] Mesh arrowMesh;
    [SerializeField] Mesh sphereMesh;

    Vector3 WindDirection;


    public void SetWindData(Vector3 dir, Color col)
    {
        if (dir.magnitude <= 0.001f)
        {
            ArrowObject.GetComponent<MeshFilter>().mesh = sphereMesh;
            ArrowObject.transform.localScale = new Vector3(0.1f,0.1f,0.1f);
        }
        else
        {
            ArrowObject.GetComponent<MeshFilter>().mesh = arrowMesh;
            ArrowObject.transform.localScale = new Vector3 (30.0f, 30.0f, 30.0f);

            WindDirection = dir;
            if (dir.magnitude > 0.001f)
                ArrowObject.transform.rotation = Quaternion.LookRotation(WindDirection, Vector3.up);
            ForceField.directionX = dir.x;
            ForceField.directionY = dir.y;
            ForceField.directionZ = dir.z;
        }

        ArrowObject.GetComponent<MeshRenderer>().material.color = col;
    }
}
