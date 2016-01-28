package DataStructures;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jeigen.DenseMatrix;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import Utils.Constants;
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
			SE3 camToWorld = camToWorldSim3.getSE3();
			DenseMatrix rotation = camToWorld.getRotationMat().div(camToWorldSim3.getScale());
			DenseMatrix translation = camToWorld.getTranslationMat();
			
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
	
}
