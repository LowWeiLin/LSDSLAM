package DataStructures;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.Random;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;

public class TrackingReference {

	public Frame keyframe;
	
	
	
	// Array of vector3, for each pyramid level.
	public jeigen.DenseMatrix[][] pointCloudLvl = new jeigen.DenseMatrix[Constants.PYRAMID_LEVELS][];
	
	/**
	 * ReferenceFrame constructor
	 */
	public TrackingReference() {
	}
	
	public TrackingReference(Frame frame) {
		this.keyframe = frame;
		
		initialize();
	}
	
	public void initialize() {
		
		for (int i=0 ; i<Constants.PYRAMID_LEVELS ; i++) {
			int size = (int) keyframe.imageLvl[i].total();
			this.keyframe.inverseDepthLvl[i] = new float[size];
			this.keyframe.inverseDepthVarianceLvl[i] = new float[size];
			randomizeInverseDepth(i);
		}
	}
	
	public void randomizeInverseDepth(int level) {
		// TODO: do random
		// Set to 1s for now.
		Arrays.fill(this.keyframe.inverseDepthLvl[level], 1);
		
		Random rand = new Random();
		rand.setSeed(System.nanoTime());
		// Randomize 0.5 - 1.5
		for (int i=0 ; i<this.keyframe.inverseDepthLvl[level].length ; i++) {
			this.keyframe.inverseDepthLvl[level][i] = 0.5f + rand.nextFloat();
		}
		
		/*
		// Use tsukuba ground truth as depth.
		Mat tsukubaGroundTruth = Highgui.imread("test0.jpg");
		Imgproc.cvtColor(tsukubaGroundTruth, tsukubaGroundTruth, Imgproc.COLOR_RGB2GRAY);
		
		// Pyramid down
		for (int i=0 ; i<level ; i++) {
			Imgproc.pyrDown(tsukubaGroundTruth, tsukubaGroundTruth);
		}
		
		byte[] data = new byte[(int) tsukubaGroundTruth.total()];
		tsukubaGroundTruth.get(0, 0, data);
		for (int i=0 ; i<tsukubaGroundTruth.total() ; i++) {
			if ((255 - data[i] &0xFF) != 0) {
				this.frame.inverseDepthLvl[level][i] = 1.0f/(float)(255 - data[i] & 0xFF);
			} else {
				this.frame.inverseDepthLvl[level][i] = 0.5f + rand.nextFloat();
			}
		}
		*/

		
		// Test drawing a box
//		int index = 0;
//		for (int j=0 ; j<height() ; j++){
//			for (int i=0 ; i<width() ; i++) {
//				if (j > 100 && j < 200 && i>100 && i<200) {
//					this.frame.inverseDepth[index] = 0.5f;
//				} else {
//					this.frame.inverseDepth[index] = 1.0f;
//				}
//				index++;
//			}
//		}
		
		
		
		
		Arrays.fill(this.keyframe.inverseDepthVarianceLvl[level], Constants.VAR_RANDOM_INIT_INITIAL);//TODO: increase value?
		
		
	}
	
	/**
	 * Create 3D points from inverse depth values
	 */
	public jeigen.DenseMatrix[] createPointCloud(float[] inverseDepth,
			float[] inverseDepthVariance, int width, int height, int level) {
		
		System.out.println("createPointCloud: " + level);
		
		jeigen.DenseMatrix[] pointCloud = new jeigen.DenseMatrix[width*height];
		
		double fxInv = Constants.fxInv[level];
		double fyInv = Constants.fyInv[level];
		double cxInv = Constants.cxInv[level];
		double cyInv = Constants.cyInv[level];
		
		for (int y=1 ; y<height-1 ; y++) {
			for (int x=1 ; x<width-1 ; x++) {
				
				// Index to reference pixel
				int pixelIndex = x + y*width;
				
				// Get idepth, variance
				float idepth = inverseDepth[pixelIndex];
				float var = inverseDepthVariance[pixelIndex];
				
				// Skip if depth/variance is not valid
				if(idepth == 0 || var <= 0) {
					pointCloud[pixelIndex] = null;
					continue;
				}

				
				// Set point, calculated from inverse depth
				pointCloud[pixelIndex] = (new jeigen.DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).div(idepth);
				if (x==3 && y==3){
					System.out.println("Pt: " + (fxInv*x + cxInv) + " " + 
									   		(fyInv*y + cyInv) + " " +
									   		1);
				}
			}
		}
		
		return pointCloud;
	}
	
//	public int width() {
//		return frame.width();
//	}
//	
//	public int height() {
//		return frame.height();
//	}
	
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
