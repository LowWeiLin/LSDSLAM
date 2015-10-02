package DataStructures;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import Utils.Constants;

public class TrackingReference {

	public Frame keyframe;
	
	
	
	// Array of vector3, for each pyramid level. posData
	public jeigen.DenseMatrix[][] pointCloudLvl = new jeigen.DenseMatrix[Constants.PYRAMID_LEVELS][];
	
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
	public jeigen.DenseMatrix[] createPointCloud(float[] inverseDepth,
			float[] inverseDepthVariance, int width, int height, int level) {
		
		jeigen.DenseMatrix[] pointCloud = new jeigen.DenseMatrix[width*height];
		
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
					pointCloud[idx] = null;
					continue;
				}
				
				// Set point, calculated from inverse depth
				pointCloud[idx] = (new jeigen.DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).div(idepth);
				
			}
		}
		return pointCloud;
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

	
}
