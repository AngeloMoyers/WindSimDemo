using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class ParticleMover : MonoBehaviour
{
    [Header("TuningData")]
    [SerializeField] GameObject WindRegionObjectPrefab;

    [Header("Shaders")]
    [SerializeField] ComputeShader Navier;
    [SerializeField] ComputeShader Solver;
    [SerializeField] ComputeShader StructuredBufferToTex;
    [SerializeField] ComputeShader StructuredBufferUtil;
    [SerializeField] ComputeShader InputShader;

    [Header("Tuning Data")]
    [SerializeField] uint CanvasDimensions = 512;
    [SerializeField][Tooltip("Each axis must be a multiple of 8.")] Vector3Int SimulationDimensions = new Vector3Int(8,8,16);
    [SerializeField] uint SolverIterations = 80;
    [SerializeField] float Viscosity;
    [SerializeField] float ForceStrength = 1;
    [SerializeField] float ForceRadius = 1;
    [SerializeField] float ForceFalloff = 2;
    [SerializeField] float VelocityDissipation = -0.999f;
    [SerializeField] float ForceMultiplier = 1f;
    [SerializeField] float ImpulseMultiplier = 10f;
    [SerializeField] bool RemoveDivergence = false;

    private CommandBuffer CmdBuffer;
    private RenderTexture VisTex;

    struct KernelHandles
    {
        public int Velocity2Tex;
        public int CopyStructureBuffer;
        public int JacobiSolve;
        public int ClearStructureBuffer;
        public int Advection;
        public int Divergence;
        public int CalculateNonDivergent;
        public int AddMouseForce;
    }
    static KernelHandles Handles;

    private Vector2 mousePrevious;

    //Data Buffers
    public ComputeBuffer VelocityBuffer;                     
    public ComputeBuffer IncomingForceBuffer;                
    public ComputeBuffer DivergenceBuffer;                   
    public ComputeBuffer PressureBuffer;                     


    public static ComputeBuffer PingBuffer;                       
    public static ComputeBuffer PongBuffer;                      

    List<WindRegion> WindRegions = new List<WindRegion>();

    private void Start()
    {
        InitSimulation();
        InitWindRegions();
    }

    private void Update()
    {
        UpdateKernalParams();

        Vector4[] data = new Vector4[SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z];
        VelocityBuffer.GetData(data);

        if (data.Length > 0)
        {
            for (int i = 0; i < data.Length; i++)
            {
                WindRegions[i].SetWindData(data[i] * ImpulseMultiplier, Color.Lerp(Color.red, Color.green, data[i].magnitude));
            }
        }

        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            AddForce(new Vector3(SimulationDimensions.x / 2, SimulationDimensions.y / 2, SimulationDimensions.z - 1 ), -Vector3.forward, 10.0f);
        }

        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            AddForce(new Vector3(SimulationDimensions.x / 2, SimulationDimensions.y / 2, 0), Vector3.forward, 10.0f);
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            AddSphericalForce(new Vector3(SimulationDimensions.x / 2, SimulationDimensions.y / 2, SimulationDimensions.z / 2), 10.0f);
        }
    }

    public void AddForce(Vector3 position, Vector3 forceDirection, float strength)
    {
        int index = CoordToIndex(Vector3Int.FloorToInt(position));
        if (index >= 0 && index < SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z)
        {
            Vector4[] data = new Vector4[SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z];
            IncomingForceBuffer.GetData(data);
            Vector3 force = forceDirection * strength * ForceMultiplier;
            data[index] += new Vector4(force.x, force.y, force.z, 0.0f);
            IncomingForceBuffer.SetData(data);
        }
    }

    public void AddSphericalForce(Vector3 position, float strength)
    {
        Vector3Int centralCoord = Vector3Int.FloorToInt(position);

        for (int z = centralCoord.z - 1; z <= centralCoord.z + 1; z++)
        {
            for (int y = centralCoord.y - 1; y <= centralCoord.y + 1; y++)
            {
                for (int x = centralCoord.x - 1; x <= centralCoord.x + 1; x++)
                {
                    Vector3Int currentCoord = new Vector3Int(x, y, z);
                    if (currentCoord.Equals(centralCoord))
                        continue;

                    int currentIndex = CoordToIndex(currentCoord);
                    if (currentIndex >= 0 && currentIndex < SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z)
                    {
                        Vector4[] data = new Vector4[SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z];
                        IncomingForceBuffer.GetData(data);
                        Vector3 force = (Vector3)(currentCoord - centralCoord) * strength;

                        data[currentIndex] += new Vector4(force.x, force.y, force.z, strength);
                        IncomingForceBuffer.SetData(data);
                    }
                }
            }
        }
    }

    private int CoordToIndex(Vector3Int coord)
    {
        return (coord.z * (SimulationDimensions.x) * (SimulationDimensions.y)) + (coord.y * (SimulationDimensions.x)) + coord.x;
    }

    private void InitSimulation()
    {
        ComputeShaderUtility.Initialize();

        VisTex = new RenderTexture((int)CanvasDimensions, (int)CanvasDimensions, 0)
        {
            enableRandomWrite = true,
            useMipMap = false
        };
        VisTex.Create();

        InitHandles();

        StructuredBufferToTex.SetInt("uPressureResultsRes", (int)CanvasDimensions);
        StructuredBufferToTex.SetInt("uVelocityResultsRes", (int)CanvasDimensions);

        CmdBuffer = new CommandBuffer()
        {
            name = "CommandBuffer"
        };

        CmdBuffer.SetGlobalInt("uResolutionX", (int)SimulationDimensions.x);
        CmdBuffer.SetGlobalInt("uResolutionY", (int)SimulationDimensions.y);
        CmdBuffer.SetGlobalInt("uResolutionZ", (int)SimulationDimensions.z);
        CmdBuffer.SetGlobalFloat("uTimeStep", 1);
        CmdBuffer.SetGlobalFloat("uGridScale", 1);

        InitDataBuffers();

        CmdBuffer.BeginSample("Wind Simulation");
        AddForce(VelocityBuffer, IncomingForceBuffer);
        Diffuse(VelocityBuffer);
        Project(VelocityBuffer, DivergenceBuffer, PressureBuffer);
        Advect(VelocityBuffer, VelocityBuffer, VelocityDissipation);
        Project(VelocityBuffer, DivergenceBuffer, PressureBuffer);


        CmdBuffer.EndSample("Wind Simulation");

        Camera.main.AddCommandBuffer(CameraEvent.AfterEverything, CmdBuffer);

    }

    private void InitDataBuffers()
    {
        VelocityBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
        IncomingForceBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
        DivergenceBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
        PressureBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
        PingBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
        PongBuffer = new ComputeBuffer(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z, sizeof(float) * 4);
    }

    private void InitHandles()
    {
        Handles = new KernelHandles()
        {
            Velocity2Tex = ComputeShaderUtility.GetKernelHandle(StructuredBufferToTex, "VelocityStructureToTexture"),
            CopyStructureBuffer = ComputeShaderUtility.GetKernelHandle(StructuredBufferUtil, "CopyStructuredBuffer"),
            JacobiSolve = ComputeShaderUtility.GetKernelHandle(Solver, "JacobiSolve"),
            ClearStructureBuffer = ComputeShaderUtility.GetKernelHandle(StructuredBufferUtil, "ClearStructuredBuffer"),
            Advection = ComputeShaderUtility.GetKernelHandle(Navier, "Advection"),
            Divergence = ComputeShaderUtility.GetKernelHandle(Navier, "Divergence"),
            CalculateNonDivergent = ComputeShaderUtility.GetKernelHandle(Navier, "CalculateNonDivergent"),
            AddMouseForce = ComputeShaderUtility.GetKernelHandle(InputShader, "AddMouseForce")
        };
    }

    private void InitWindRegions()
    {
        for (int z = 0; z < SimulationDimensions.z; z++)
        {
            for (int y = 0; y < SimulationDimensions.y; y++)
            {
                for (int x = 0; x < SimulationDimensions.x; x++)
                {
                    var region = Instantiate(WindRegionObjectPrefab, new Vector3(x, y, z), Quaternion.identity, transform).GetComponent<WindRegion>();
                    WindRegions.Add(region);
                }
            }
        }
    }
    private void DrawVisual()
    {
        SetComputeBufferOnCommandBuffer(CmdBuffer, VelocityBuffer, "uVelocityBufferToTexSource");
        StructuredBufferToTex.SetTexture(Handles.Velocity2Tex, "uVelocityBufferToTexTarget", VisTex); 

        DispatchCommand(CmdBuffer, StructuredBufferToTex, Handles.Velocity2Tex, CanvasDimensions, CanvasDimensions, 1);

        CmdBuffer.Blit(VisTex, BuiltinRenderTextureType.CameraTarget);
    }

    private void UpdateKernalParams()
    {
        SetFloatOnAllShaders(Time.time, "uTime");

        float forceController = 0;

        if (Input.GetMouseButton(0)) forceController = ForceStrength;

        InputShader.SetFloat("uForceController", forceController);
        InputShader.SetFloat("uForceMultiplier", ForceMultiplier);
        InputShader.SetFloat("uForceRadius", ForceRadius);
        InputShader.SetFloat("uForceFalloff", ForceFalloff);

        Vector2 mousePos = MousePos();

        InputShader.SetVector("uMouseCurrent", mousePos);          
        InputShader.SetVector("uMousePrev", mousePrevious);                    

        mousePrevious = mousePos;
    }

    private Vector2 MousePos()
    {
        Vector3 mousePixelPos = Input.mousePosition;
        Vector2 mousePosNormalized = Camera.main.ScreenToViewportPoint(mousePixelPos);
        mousePosNormalized = new Vector2(Mathf.Clamp01(mousePosNormalized.x), Mathf.Clamp01(mousePosNormalized.y));
        return new Vector2(mousePosNormalized.x * SimulationDimensions.x, mousePosNormalized.y * SimulationDimensions.y);
    }

    private void SetFloatOnAllShaders(float val, string name)
    {
        Navier.SetFloat(name, val);
        Solver.SetFloat(name, val);
        StructuredBufferToTex.SetFloat(name, val);
        StructuredBufferUtil.SetFloat(name, val);
    }

    private void ReleaseDataBuffers()
    {
        VelocityBuffer.Release();                       
        DivergenceBuffer.Release();                       
        PressureBuffer.Release();                       
        PingBuffer.Release();                      
        PongBuffer.Release();
    }

    private void Release()
    {
        VisTex.Release();
        ComputeShaderUtility.Release();
        ReleaseDataBuffers();
    }

    private void OnDisable()
    {
        Release();
    }

    private void SetComputeBufferOnCommandBuffer(CommandBuffer command, ComputeBuffer compute, string bufferName)
    {
        command.SetGlobalBuffer(bufferName, compute);
    }

    private void DispatchCommand(CommandBuffer command, ComputeShader shader, int kernel, uint threadX, uint threadY, uint threadZ)
    {
        DispatchDimensions dims = ComputeShaderUtility.CheckGetDispatchDimensions(shader, kernel, threadX, threadY, threadZ);


        if ((int)dims.dispatchX == 0 || (int)dims.dispatchY == 0 || (int)dims.dispatchZ == 0)
        {
            Debug.Log("Dimension is zero on: " + shader.name);
        }
        else
        {
            command.DispatchCompute(shader, kernel, (int)dims.dispatchX, (int)dims.dispatchY, (int)dims.dispatchZ);
        }
    }

    private void ClearBuffer(ComputeBuffer buffer, Vector4 clearVal)
    {
        CmdBuffer.SetGlobalVector("uClearValueStructuredBuffer", new Vector4(0.0f, 0.0f, 0.0f, 0.0f));
        SetComputeBufferOnCommandBuffer(CmdBuffer, buffer, "uClearTargetStructuredBuffer");
        DispatchCommand(CmdBuffer, StructuredBufferUtil, Handles.ClearStructureBuffer, (uint)(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z), 1, 1);
    }

    private void AddForce(ComputeBuffer velocityBuffer, ComputeBuffer incomingForcebuffer)
    {
        SetComputeBufferOnCommandBuffer(CmdBuffer, velocityBuffer, "uAppliedForce");
        SetComputeBufferOnCommandBuffer(CmdBuffer, incomingForcebuffer, "uIncomingForce");
        DispatchCommand(CmdBuffer, InputShader, Handles.AddMouseForce, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);
    }

    private void Diffuse(ComputeBuffer buffer)
    {
        if (Viscosity <= 0) return;
        if (!PingBuffer.IsValid() || !PongBuffer.IsValid()) return;

        float centerFactor = 1.0f / (Viscosity * 1);
        float reciprocalDiagonal = (Viscosity * 1.0f) / (1.0f + 6.0f * (Viscosity * 1.0f));

        CmdBuffer.SetGlobalFloat("uCenterFactor", centerFactor);
        CmdBuffer.SetGlobalFloat("uReciprocalDiagonal", reciprocalDiagonal);

        bool pingPong = false;

        for (int i = 0; i < SolverIterations; i++)
        {
            pingPong = !pingPong;
            if (pingPong)
            {
                SetComputeBufferOnCommandBuffer(CmdBuffer, buffer, "uTarget");
                SetComputeBufferOnCommandBuffer(CmdBuffer, buffer, "uUpdatedBuffer");
                SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uResults");
            }
            else
            {
                SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uTarget");
                SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uUpdatedBuffer");
                SetComputeBufferOnCommandBuffer(CmdBuffer, buffer, "uResults");
            }

            CmdBuffer.SetGlobalInt("uCurrentIteration", i);
            DispatchCommand(CmdBuffer, Solver, Handles.JacobiSolve, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);
        }

        if (pingPong)
        {
            SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uCopySource");
            SetComputeBufferOnCommandBuffer(CmdBuffer, buffer, "uCopyTarget");
            DispatchCommand(CmdBuffer, StructuredBufferUtil, Handles.CopyStructureBuffer, (uint)(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z), 1,  1);
        }

        ClearBuffer(PingBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    private void CalculateFieldDivergence(ComputeBuffer target, ComputeBuffer divergence)
    {
        SetComputeBufferOnCommandBuffer(CmdBuffer, target, "uDivergenceVectorField");
        SetComputeBufferOnCommandBuffer(CmdBuffer, divergence, "uDivergenceValues");
        DispatchCommand(CmdBuffer, Navier, Handles.Divergence, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);
    }

    private void CalculateNonDivergentFromPressure(ComputeBuffer nonZero, ComputeBuffer pressure, ComputeBuffer nonDivergent)
    {
        SetComputeBufferOnCommandBuffer(CmdBuffer, nonZero, "uNonZeroDivergenceVelocity");
        SetComputeBufferOnCommandBuffer(CmdBuffer, pressure, "uPressureField");
        SetComputeBufferOnCommandBuffer(CmdBuffer, nonDivergent, "uNonDivergent");

        DispatchCommand(CmdBuffer, Navier, Handles.CalculateNonDivergent, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);
    }

    private void Project(ComputeBuffer targetBuffer, ComputeBuffer divergenceBuffer, ComputeBuffer pressureBuffer)
    {
        if (!PingBuffer.IsValid() || !PongBuffer.IsValid()) return;

        if (RemoveDivergence)
            CalculateFieldDivergence(targetBuffer, divergenceBuffer);

        float centerFactor = -1.0f * 1 * 1;
        float diagonalFactor = 0.16f;

        CmdBuffer.SetGlobalFloat("uCenterFactor", centerFactor);
        CmdBuffer.SetGlobalFloat("uReciprocalDiagonal", diagonalFactor);

        if (RemoveDivergence)
            SetComputeBufferOnCommandBuffer(CmdBuffer, divergenceBuffer, "uTarget");

        ClearBuffer(pressureBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));

        bool pingPong = false;

        for (int i = 0; i < SolverIterations; i++)
        {
            pingPong = !pingPong;
            if (pingPong)
            {
                SetComputeBufferOnCommandBuffer(CmdBuffer, pressureBuffer, "uUpdatedBuffer");
                SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uResults");
            }
            else
            {
                SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uUpdatedBuffer");
                SetComputeBufferOnCommandBuffer(CmdBuffer, pressureBuffer, "uResults");
            }

            CmdBuffer.SetGlobalInt("uCurrentIteration", i);
            DispatchCommand(CmdBuffer, Solver, Handles.CopyStructureBuffer, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);
        }

        if (pingPong)
        {
            SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uCopySource");
            SetComputeBufferOnCommandBuffer(CmdBuffer, pressureBuffer, "uCopyTarget");
            DispatchCommand(CmdBuffer, StructuredBufferUtil, Handles.CopyStructureBuffer, (uint)(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z), 1, 1);
        }

        ClearBuffer(PingBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));

        if (RemoveDivergence)
        {
            CalculateNonDivergentFromPressure(targetBuffer, pressureBuffer, PingBuffer);

            SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uCopySource");
            SetComputeBufferOnCommandBuffer(CmdBuffer, targetBuffer, "uCopyTarget");
            DispatchCommand(CmdBuffer, StructuredBufferUtil, Handles.CopyStructureBuffer, (uint)(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z), 1, 1);
        }

        ClearBuffer(PingBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));
        ClearBuffer(PongBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    private void Advect(ComputeBuffer target, ComputeBuffer velocityBuffer, float dissipation)
    {
        CmdBuffer.SetGlobalFloat("uDissipation", dissipation);

        SetComputeBufferOnCommandBuffer(CmdBuffer, velocityBuffer, "uVelocityBuffer");
        SetComputeBufferOnCommandBuffer(CmdBuffer, target, "uAdvectTargetBuffer");
        SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uAdvectedBuffer");

        DispatchCommand(CmdBuffer, Navier, Handles.Advection, (uint)SimulationDimensions.x, (uint)SimulationDimensions.y, (uint)SimulationDimensions.z);

        SetComputeBufferOnCommandBuffer(CmdBuffer, PingBuffer, "uCopySource");
        SetComputeBufferOnCommandBuffer(CmdBuffer, target, "uCopyTarget");

        DispatchCommand(CmdBuffer, StructuredBufferUtil, Handles.CopyStructureBuffer, (uint)(SimulationDimensions.x * SimulationDimensions.y * SimulationDimensions.z), 1, 1);

        ClearBuffer(PingBuffer, new Vector4(0.0f, 0.0f, 0.0f, 0.0f));
    }

}
