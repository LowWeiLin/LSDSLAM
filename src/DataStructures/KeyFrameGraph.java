package DataStructures;

import g2o.g2o_SparseOptimizer;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jeigen.DenseMatrix;
import LieAlgebra.SIM3;
import Utils.Constants;
import Utils.PQueue;
import Utils.PlyWriter;

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

	public Map<Integer, Frame> idToKeyFrame = new HashMap<Integer, Frame>();
	

	// contains ALL edges, as soon as they are created
	List<KFConstraintStruct> edgesAll = new ArrayList<KFConstraintStruct>();

	
	// contains ALL frame poses, chronologically, as soon as they are tracked.
	// the corresponding frame may have been removed / deleted in the meantime.
	// these are the ones that are also referenced by the corresponding Frame / Keyframe object
	public ArrayList<FramePoseStruct> allFramePoses =  new ArrayList<FramePoseStruct>();


	// contains all keyframes in graph, in some arbitrary (random) order. if a frame is re-tracked,
	// it is put to the end of this list; frames for re-tracking are always chosen from the first third of
	// this list.
	//std::deque<Frame*> keyframesForRetrack;


	/** Pose graph representation in g2o */
	g2o_SparseOptimizer graph;
	
	List<Frame> newKeyframesBuffer = new ArrayList<Frame>();
	List<KFConstraintStruct> newEdgeBuffer = new ArrayList<KFConstraintStruct>();

	public int nextEdgeId;

	

	public void addFrame(Frame frame) {
	
		frame.pose.isRegisteredToGraph = true;
		FramePoseStruct pose = frame.pose;
		allFramePoses.add(pose);
		
	}
	
	public void addKeyFrame(Frame frame) {
		System.err.println("Add keyframe " + frame.id);
		if(frame.pose.graphVertex != null)
			return;

		// Insert vertex into g2o graph
		VertexSim3 vertex = new VertexSim3();
		vertex.setId(frame.id());

		SIM3 camToWorld_estimate = frame.getScaledCamToWorld();

		if(!frame.hasTrackingParent())
			vertex.setFixed(true);

		vertex.setEstimate(camToWorld_estimate);
		vertex.setMarginalized(false);

		frame.pose.graphVertex = vertex;

		newKeyframesBuffer.add(frame);

	}
	

	public void insertConstraint(KFConstraintStruct constraint)
	{
		EdgeSim3 edge = new EdgeSim3();
		edge.setId(nextEdgeId);
		++ nextEdgeId;
	
		totalEdges++;
	
		edge.setMeasurement(constraint.secondToFirst);
		edge.setInformation(constraint.information);
		edge.setRobustKernel(constraint.robustKernel);
	
		System.err.println("Constraint: " + constraint.firstFrame.id() + " - " + constraint.secondFrame.id());
		
		if (constraint.firstFrame.pose.graphVertex != null)
			System.err.println(constraint.firstFrame.id() + " - " + constraint.firstFrame.pose.graphVertex.id);
		if (constraint.secondFrame.pose.graphVertex != null)
			System.err.println(constraint.secondFrame.id() + " - " + constraint.secondFrame.pose.graphVertex.id);
		
		
		edge.resize(2);
		assert(constraint.firstFrame.pose.graphVertex != null);
		edge.setVertex(0, constraint.firstFrame.pose.graphVertex);
		assert(constraint.secondFrame.pose.graphVertex != null);
		edge.setVertex(1, constraint.secondFrame.pose.graphVertex);
	
		constraint.edge = edge;
		newEdgeBuffer.add(constraint);
	
	
		constraint.firstFrame.neighbors.add(constraint.secondFrame);
		constraint.secondFrame.neighbors.add(constraint.firstFrame);
	
		/*
		for(int i=0;i<totalVertices;i++)
		{
			//shortestDistancesMap
		}*/
	
	
	
		//edgesListsMutex.lock();
		constraint.idxInAllEdges = edgesAll.size();
		edgesAll.add(constraint);
		//edgesListsMutex.unlock();
	}
	
	public void writePointCloudToFile(String filename
			) throws FileNotFoundException, UnsupportedEncodingException {
		
		List<DenseMatrix> allPoints = new ArrayList<DenseMatrix>();
		
		for (int f=0 ; f<keyframesAll.size() ; f++) {
			
			
			Frame keyframe = keyframesAll.get(f);
			System.out.println("WRITING KF "+keyframe.id);
			
			int width = keyframe.width(0);
			int height = keyframe.height(0);
			int size = width*height;

			float[] image = keyframe.imageArrayLvl[0];
			float[] inverseDepth = keyframe.inverseDepthLvl[0];
			float[] inverseDepthVariance = keyframe.inverseDepthVarianceLvl[0];
			
			// To store position of points
			jeigen.DenseMatrix[] posData = new jeigen.DenseMatrix[size];
			jeigen.DenseMatrix[] colorAndVarData = new jeigen.DenseMatrix[size];
			
			double fxInv = Constants.fxInv[0];
			double fyInv = Constants.fyInv[0];
			double cxInv = Constants.cxInv[0];
			double cyInv = Constants.cyInv[0];
			
			// For transformation
			SIM3 camToWorldSim3 = keyframe.getScaledCamToWorld();
			
			for (int x=1 ; x<width-1 ; x++) {
				for (int y=1 ; y<height-1 ; y++) {
					
					// Index to reference pixel
					int idx = x + y*width;
					
					// Get idepth, variance
					float idepth = inverseDepth[idx];
					float var = inverseDepthVariance[idx];

					// Skip if depth/variance is not valid
					if(idepth <= 0 || var <= 0) {
						posData[idx] = null;
						continue;
					}
					
					float color = image[idx];
					float depth = (float) (1.0/idepth);
					
					// Set point position, calculated from inverse depth
					posData[idx] = (new jeigen.DenseMatrix(
							new double[][]{{fxInv*x + cxInv},
										   {fyInv*y + cyInv},
										   {1}})).mul(depth);
					
					
					// Transform to world coordinates, based on first KF
					posData[idx] = camToWorldSim3.mul(posData[idx]);
					
					
					colorAndVarData[idx] = new jeigen.DenseMatrix(
								new double[][]{{color},
												{var}});
					
					
				}
			}

			// Add tracking camera points
			for (int i=1 ; i<keyframe.trackedOnPoses.size() ; i++) {
				DenseMatrix cameraPoint = keyframe.trackedOnPoses.get(i).translation;
				//cameraPoint =  rotation.mmul(cameraPoint).add(translation);
				cameraPoint = camToWorldSim3.mul(cameraPoint);
				allPoints.add(new DenseMatrix(
						new double[][]{{cameraPoint.get(0, 0)},
									   {cameraPoint.get(1, 0)},
									   {cameraPoint.get(2, 0)},
									   {(i == keyframe.trackedOnPoses.size()-1)?0:255},
									   {f%2==0?255:0},
									   {f%2==0?0:255}})); // color based on KF
				
			}

			// Add KF camera point
			DenseMatrix cameraPoint = new DenseMatrix(new double[][]{{0},{0},{0}});
			//cameraPoint =  rotation.mmul(cameraPoint).add(translation);
			cameraPoint = camToWorldSim3.mul(cameraPoint);
			allPoints.add(new DenseMatrix(
					new double[][]{{cameraPoint.get(0, 0)},
								   {cameraPoint.get(1, 0)},
								   {cameraPoint.get(2, 0)},
								   {255},
								   {0},
								   {0}}));

			// Add points to list
			// TODO: Add color data?
			for (int i=0 ; i<size ; i++) {

				if (keyframesAll.size() > 5 && f < 5) {
					// Skip merging first few KFs, since they may not be accurate yet.
					break;
				}
				
				if (posData[i] == null) {
					continue;
				}
				double color = colorAndVarData[i].get(0, 0);
				allPoints.add(posData[i].concatDown(new DenseMatrix(
						new double[][]{{color},{color},{color}})));
	 		}
		
		}
		
		
		// Write to file
		PlyWriter.writePoints(filename, allPoints);
	}
	
	
	public void calculateGraphDistancesToFrame(Frame startFrame, Map<Frame, Integer> distanceMap)
	{
		distanceMap.put(startFrame, 0);
		
		
		PQueue<Frame> priorityQueue = new PQueue<Frame>();
		priorityQueue.add(0, startFrame);
		
		
		while (! priorityQueue.isEmpty())
		{
			
			int length = priorityQueue.peekPriority();
			Frame frame = priorityQueue.get();
			
			Integer mapEntry = distanceMap.get(frame);
			
			if (mapEntry != null && length > mapEntry)
			{
				continue;
			}
			
			for (Frame neighbor : frame.neighbors)
			{
				Integer neighborMapEntry = distanceMap.get(neighbor);
				
				if (neighborMapEntry != null && length + 1 >= neighborMapEntry) {
					continue;
				}
				if (neighborMapEntry != null) {
					neighborMapEntry = length + 1;
				} else {
					distanceMap.put(neighbor, length + 1);
				}
				priorityQueue.add(length + 1, neighbor);
			}
			
		}
		
	}
}
