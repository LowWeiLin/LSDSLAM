package DataStructures;

import LieAlgebra.SIM3;

public class FramePoseStruct {

	// parent, the frame originally tracked on. never changes.
	public FramePoseStruct trackingParent;

	// set initially as tracking result (then it's a SE(3)),
	// and is changed only once, when the frame becomes a KF (->rescale).
	public SIM3 thisToParent_raw;

	public int frameID;
	Frame frame;

	// whether this poseStruct is registered in the Graph. if true MEMORY WILL BE HANDLED BY GRAPH
	boolean isRegisteredToGraph;

	// whether pose is optimized (true only for KF, after first applyPoseGraphOptResult())
	boolean isOptimized;

	// true as soon as the vertex is added to the g2o graph.
	public boolean isInGraph;

	// graphVertex (if the frame has one, i.e. is a KF and has been added to the graph, otherwise 0).
	public VertexSim3 graphVertex;
	
	int cacheValidFor;
	static int cacheValidCounter;

	// absolute position (camToWorld).
	// can change when optimization offset is merged.
	SIM3 camToWorld;

	// new, optimized absolute position. is added on mergeOptimization.
	SIM3 camToWorld_new;

	// whether camToWorld_new is newer than camToWorld
	boolean hasUnmergedPose;
	
	
	
	
	
	
	public FramePoseStruct(Frame frame) {
		cacheValidFor = -1;
		isOptimized = false;
		thisToParent_raw = camToWorld = camToWorld_new = new SIM3();
		this.frame = frame;
		frameID = frame.id();
		trackingParent = null;
		isRegisteredToGraph = false;
		hasUnmergedPose = false;
		isInGraph = false;

		//TODO: this.graphVertex = null;

	}

	public void setPoseGraphOptResult(SIM3 camToWorld) {
		
	}
	
	public void applyPoseGraphOptResult() {
		
	}
	

	public SIM3 getCamToWorld() {
		return getCamToWorld(0);
	}
	
	public SIM3 getCamToWorld(int recursionDepth) {
		// prevent stack overflow
		assert(recursionDepth < 5000);

		// if the node is in the graph, it's absolute pose is only changed by optimization.
		//if(isOptimized) return camToWorld;


		/*
		
		// return chached pose, if still valid.
		if(cacheValidFor == cacheValidCounter)
			return camToWorld;

		// abs. pose is computed from the parent's abs. pose, and cached.
		cacheValidFor = cacheValidCounter;
		*/
		

		// return id if there is no parent (very first frame)
		if(trackingParent == null)
			return camToWorld = new SIM3();
		
		/*
		System.out.println("+++");
		System.out.println(trackingParent.getCamToWorld(recursionDepth+1).toString());
		System.out.println(thisToParent_raw.toString());
		*/
		
		
		return camToWorld = trackingParent.getCamToWorld(recursionDepth+1).mul(thisToParent_raw);
	}
	
	public void invalidateCache()
	{
		cacheValidFor = -1;
	}
}
