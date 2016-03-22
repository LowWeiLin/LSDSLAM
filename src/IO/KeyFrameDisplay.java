package IO;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jeigen.DenseMatrix;
import DataStructures.Frame;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import Utils.Constants;
import Utils.PlyWriter;


public class KeyFrameDisplay {
	
	public Frame keyframe;
	
	public SIM3 camToWorld;
	
	
	public KeyFrameDisplay(Frame keyframe) {
		this.keyframe = keyframe;
		this.camToWorld = new SIM3();
	}
	
	public List<DenseMatrix> getPointCloud(int level) {

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
				
				posData[idx] = new DenseMatrix(
						new double[][]{{(fxInv*x + cxInv)/idepth},
									   {(fyInv*y + cyInv)/idepth},
									   {1.0/idepth}});
				
				// Transform
				posData[idx] = camToWorld.mul(posData[idx]);
				
				colorAndVarData[idx] = new DenseMatrix(
							new double[][]{{color},
											{var}});
				
			}
		}
		
		// Return points
		return Arrays.asList(posData);
	}
	
	public void writePointCloudToFile(String filename) throws FileNotFoundException, UnsupportedEncodingException {
		writePointCloudToFile(filename, getPointCloud(1),keyframe.width(1), keyframe.height(1));
	}
	
	public void writePointCloudToFile(String filename, List<DenseMatrix> list, int width, int height) throws FileNotFoundException, UnsupportedEncodingException {
		
		List<DenseMatrix> cameraPosePoints = generateCameraPosePoints();
		List<DenseMatrix> allPoints = new ArrayList<DenseMatrix>();
		
		// Insert point cloud points
		int size = width*height;
		for (int i=0 ; i<size ; i++) {
			if (list.get(i) == null) {
				continue;
			}
			allPoints.add(list.get(i));
		}
		
		// Insert camera points
		for (int i=0 ; i<cameraPosePoints.size() ; i++) {
			DenseMatrix point = cameraPosePoints.get(i);
			allPoints.add(point);
 		}
		
		// Write to file
		PlyWriter.writePoints(filename, allPoints);
		
	}

	public List<DenseMatrix> generateCameraPosePoints() {
		List<SE3> cameraPose = keyframe.trackedOnPoses;
		
		List<DenseMatrix> cameraPoints = new ArrayList<>();
		
		DenseMatrix pt;
		
		double camSize = 0.03;
		
		// For KF, Plot the XYZ axis
		// Center
		pt = new DenseMatrix(new double[][]{{0},{0},{0}});
		pt = camToWorld.mul(pt);
		DenseMatrix c = new DenseMatrix(
				new double[][]{{pt.get(0, 0)},{pt.get(1, 0)},{pt.get(2, 0)},
							   {255},{255},{0}});
		cameraPoints.add(c);
		
		// X
		pt = new DenseMatrix(new double[][]{{camSize},{0},{0}});
		pt = camToWorld.mul(pt);
		DenseMatrix x = new DenseMatrix(
				new double[][]{{pt.get(0, 0)},{pt.get(1, 0)},{pt.get(2, 0)},
							   {255},{0},{0}});
		cameraPoints.add(x);
		
		// Y
		pt = new DenseMatrix(new double[][]{{0},{camSize},{0}});
		pt = camToWorld.mul(pt);
		DenseMatrix y = new DenseMatrix(
				new double[][]{{pt.get(0, 0)},{pt.get(1, 0)},{pt.get(2, 0)},
							   {0},{255},{0}});
		cameraPoints.add(y);
		
		// Z
		pt = new DenseMatrix(new double[][]{{0},{0},{camSize}});
		pt = camToWorld.mul(pt);
		DenseMatrix z = new DenseMatrix(
				new double[][]{{pt.get(0, 0)},{pt.get(1, 0)},{pt.get(2, 0)},
							   {0},{0},{255}});
		cameraPoints.add(z);
		
		for (int i=0 ; i<cameraPose.size() ; i++) {
			
			SE3 se3 = cameraPose.get(i);
			
			pt = new DenseMatrix(new double[][]{{se3.translation.get(0, 0)},
								   {se3.translation.get(1, 0)},
								   {se3.translation.get(2, 0)}});

			pt = camToWorld.mul(pt);
			
			DenseMatrix point;
			if (i == 0) {
				point = new DenseMatrix(
						new double[][]{{pt.get(0, 0)},
									   {pt.get(1, 0)},
									   {pt.get(2, 0)},
									   {255},{255},{0}});
				
			} else {
				point = new DenseMatrix(
					new double[][]{{pt.get(0, 0)},
								   {pt.get(1, 0)},
								   {pt.get(2, 0)},
								   {255},{0},{0}});
			}
			
			cameraPoints.add(point);
			
			
		}
		
		return cameraPoints;
	}
	
}
