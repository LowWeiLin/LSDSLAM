package DataStructures;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import Utils.Constants;

public class TrackingReference {

	public Frame keyframe;
	
	
	
	// Array of vector3, for each pyramid level. posData, makes up the point cloud.
	public jeigen.DenseMatrix[][] posDataLvl = new jeigen.DenseMatrix[Constants.PYRAMID_LEVELS][];
	public jeigen.DenseMatrix[][] colorAndVarDataLvl = new jeigen.DenseMatrix[Constants.PYRAMID_LEVELS][];
	
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
		
		jeigen.DenseMatrix[] posData = new jeigen.DenseMatrix[width*height];
		jeigen.DenseMatrix[] colorAndVarData = new jeigen.DenseMatrix[width*height];
		
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
				
				float depth = (float) (1.0/idepth);
				
				// Set point, calculated from inverse depth
				
				posData[idx] = (new jeigen.DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).div(idepth);
				
				/*
				posData[idx] = (new jeigen.DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).mul(depth);
				*/
				colorAndVarData[idx] = new jeigen.DenseMatrix(
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
	
	public static void writePointCloudToFile(String filename, jeigen.DenseMatrix[] pointCloud, int width, int height) throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(filename, "ASCII");
	
		writer.println(3);
		
		int size = width*height;
		for (int i=0 ; i<size ; i++) {		
			if (pointCloud[i] == null) {
				continue;
			}
			
			writer.printf("%.6f ", pointCloud[i].get(0, 0));
			writer.printf("%.6f ", pointCloud[i].get(1, 0));
			writer.printf("%.6f\n", pointCloud[i].get(2, 0));
			
 		}
		
		writer.close();
	}

	public void importFrame(Frame currentKeyFrame) {
		keyframe = currentKeyFrame;
	}

	public void invalidate() {
		keyframe = null;
	}
	
}
