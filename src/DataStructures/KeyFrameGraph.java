package DataStructures;

import java.util.ArrayList;

import LieAlgebra.SIM3;

public class KeyFrameGraph {


	public int totalPoints;
	public int totalEdges;
	public int totalVertices;


	//=========================== Keyframe & Posen Lists & Maps ====================================

	// contains ALL keyframes, as soon as they are "finished".
	// does NOT yet contain the keyframe that is currently being created.
	public ArrayList<Frame> keyframesAll = new ArrayList<Frame>();


	/** Maps frame ids to keyframes. Contains ALL Keyframes allocated, including the one that currently being created. */
	/* this is where the shared pointers of Keyframe Frames are kept, so they are not deleted ever */
	//std::unordered_map< int, std::shared_ptr<Frame>, std::hash<int>, std::equal_to<int>,
	//Eigen::aligned_allocator< std::pair<const int, std::shared_ptr<Frame> > > > idToKeyFrame;


	// contains ALL edges, as soon as they are created
	//std::vector< KFConstraintStruct*, Eigen::aligned_allocator<KFConstraintStruct*> > edgesAll;



	// contains ALL frame poses, chronologically, as soon as they are tracked.
	// the corresponding frame may have been removed / deleted in the meantime.
	// these are the ones that are also referenced by the corresponding Frame / Keyframe object
	public ArrayList<FramePoseStruct> allFramePoses =  new ArrayList<FramePoseStruct>();


	// contains all keyframes in graph, in some arbitrary (random) order. if a frame is re-tracked,
	// it is put to the end of this list; frames for re-tracking are always chosen from the first third of
	// this list.
	//std::deque<Frame*> keyframesForRetrack;


	/** Pose graph representation in g2o */
	//g2o::SparseOptimizer graph;
	
	//std::vector< Frame*, Eigen::aligned_allocator<Frame*> > newKeyframesBuffer;
	//std::vector< KFConstraintStruct*, Eigen::aligned_allocator<FramePoseStruct*> > newEdgeBuffer;

	public int nextEdgeId;

	

	public void addFrame(Frame frame) {
	
		frame.pose.isRegisteredToGraph = true;
		FramePoseStruct pose = frame.pose;
		allFramePoses.add(pose);
		
	}
	
	
}
