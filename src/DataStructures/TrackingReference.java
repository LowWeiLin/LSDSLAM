package DataStructures;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import jeigen.DenseMatrix;
import LieAlgebra.SE3;
import Utils.Constants;
import Utils.PlyWriter;

public class TrackingReference {

	public Frame keyframe;
	
	// Array of vector3, for each pyramid level. posData, makes up the point cloud.
	public DenseMatrix[][] posDataLvl = new DenseMatrix[Constants.PYRAMID_LEVELS][];
	public DenseMatrix[][] colorAndVarDataLvl = new DenseMatrix[Constants.PYRAMID_LEVELS][];
	
	/**
	 * ReferenceFrame constructor
	 */
	public TrackingReference() {
	}
	
	public TrackingReference(Frame frame) {
		this.keyframe = frame;
	}
	
	/**
	 * Create 3D points from inverse depth values
	 */
	public void createPointCloud(int level) {
		
		
		int width = keyframe.width(level);
		int height = keyframe.height(level);

		float[] image = keyframe.imageArrayLvl[level];
		float[] inverseDepth = keyframe.inverseDepthLvl[level];
		float[] inverseDepthVariance = keyframe.inverseDepthVarianceLvl[level];
		
		DenseMatrix[] posData = new DenseMatrix[width*height];
		DenseMatrix[] colorAndVarData = new DenseMatrix[width*height];
		
		double fxInv = Constants.fxInv[level];
		double fyInv = Constants.fyInv[level];
		double cxInv = Constants.cxInv[level];
		double cyInv = Constants.cyInv[level];
		
		for (int x=1 ; x<width-1 ; x++) {
			for (int y=1 ; y<height-1 ; y++) {
				
				// Index to reference pixel
				int idx = x + y*width;
				
				// Get idepth, variance
				float idepth = inverseDepth[idx];
				float var = inverseDepthVariance[idx];

				// Skip if depth/variance is not valid
				if(idepth == 0 || var <= 0) {
					posData[idx] = null;
					continue;
				}
				
				float color = image[idx];
				
				// Set point, calculated from inverse depth
				
				posData[idx] = (new DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).div(idepth);
				
				/*
				float depth = (float) (1.0/idepth);
				posData[idx] = (new DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).mul(depth);
				*/
				colorAndVarData[idx] = new DenseMatrix(
							new double[][]{{color},
											{var}});
				
			}
		}
		
		posDataLvl[level] = posData;
		colorAndVarDataLvl[level] = colorAndVarData;
		
	}
	
	public int width(int level) {
		return keyframe.width(level);
	}
	
	public int height(int level) {
		return keyframe.height(level);
	}
	
	public void writePointCloudToFile(String filename, DenseMatrix[] pointCloud, int width, int height) throws FileNotFoundException, UnsupportedEncodingException {
		
		List<DenseMatrix> cameraPosePoints = generateCameraPosePoints();
		List<DenseMatrix> allPoints = new ArrayList<DenseMatrix>();
		
		// Insert point cloud points
		int size = width*height;
		for (int i=0 ; i<size ; i++) {
			if (pointCloud[i] == null) {
				continue;
			}
			allPoints.add(pointCloud[i]);
		}
		
		// Insert camera points
		for (int i=0 ; i<cameraPosePoints.size() ; i++) {
			DenseMatrix point = cameraPosePoints.get(i);
			allPoints.add(point);
 		}
		
		// Write to file
		PlyWriter.writePoints(filename, allPoints);
		
	}

	public void importFrame(Frame currentKeyFrame) {
		keyframe = currentKeyFrame;
	}

	public void invalidate() {
		keyframe = null;
	}
	
	public List<DenseMatrix> generateCameraPosePoints() {
		List<SE3> cameraPose = keyframe.trackedOnPoses;
		
		List<DenseMatrix> cameraPoints = new ArrayList<>();
		
		for (int i=0 ; i<cameraPose.size() ; i++) {
			SE3 se3 = cameraPose.get(i);
			DenseMatrix point = new DenseMatrix(
					new double[][]{{se3.translation.get(0, 0)},
								   {se3.translation.get(1, 0)},
								   {se3.translation.get(2, 0)},
								   {255},
								   {255},
								   {0}});
			cameraPoints.add(point);
			
			// For KF, plot the XYZ axis
			if (i==0) {
				DenseMatrix x = new DenseMatrix(
						new double[][]{{0.1},{0},{0},
									   {255},{0},{0}});
				cameraPoints.add(x);
				DenseMatrix y = new DenseMatrix(
						new double[][]{{0},{0.1},{0},
									   {0},{255},{0}});
				cameraPoints.add(y);
				DenseMatrix z = new DenseMatrix(
						new double[][]{{0},{0},{0.1},
									   {0},{0},{255}});
				cameraPoints.add(z);
			}
		}
		
		return cameraPoints;
	}
	
}
