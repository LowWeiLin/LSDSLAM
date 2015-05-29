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

public class ReferenceFrame {

	public Frame frame;
	
	public float[] inverseDepth;
	public float[] inverseDepthVariance;
	
	
	// Array of vector3
	public jeigen.DenseMatrix[] pointCloud;
	
	
	/**
	 * ReferenceFrame constructor
	 */
	public ReferenceFrame(Frame frame) {
		this.frame = frame;
		
		initialize();
	}
	
	public void initialize() {
		int size = (int) frame.image.total();
		inverseDepth = new float[size];
		inverseDepthVariance = new float[size];
		randomizeInverseDepth();
		
	}
	
	public void randomizeInverseDepth() {
		// TODO: do random
		// Set to 1s for now.
		Arrays.fill(inverseDepth, 1);
		
		Random rand = new Random();
		rand.setSeed(System.nanoTime());
		// Randomize 0.5 - 1.5
//		for (int i=0 ; i<inverseDepth.length ; i++) {
//			inverseDepth[i] = 0.5f + rand.nextFloat();
//		}
		
		
		// Use tsukuba ground truth as depth.
		Mat tsukubaGroundTruth = Highgui.imread("test0.jpg");
		Imgproc.cvtColor(tsukubaGroundTruth, tsukubaGroundTruth, Imgproc.COLOR_RGB2GRAY);
		byte[] data = new byte[(int) tsukubaGroundTruth.total()];
		tsukubaGroundTruth.get(0, 0, data);
		for (int i=0 ; i<tsukubaGroundTruth.total() ; i++) {
			if ((255 - data[i] &0xFF) != 0) {
				inverseDepth[i] = 1.0f/(float)(255 - data[i] & 0xFF);
			} else {
				inverseDepth[i] = 0.5f + rand.nextFloat();
			}
		}
		

		
		// Test drawing a box
//		int index = 0;
//		for (int j=0 ; j<height() ; j++){
//			for (int i=0 ; i<width() ; i++) {
//				if (j > 100 && j < 200 && i>100 && i<200) {
//					inverseDepth[index] = 0.5f;
//				} else {
//					inverseDepth[index] = 1.0f;
//				}
//				index++;
//			}
//		}
		
		
		
		
		Arrays.fill(inverseDepthVariance, Constants.VAR_RANDOM_INIT_INITIAL);//TODO: increase value?
		
		
	}
	
	/**
	 * Create 3D points from inverse depth values
	 */
	public jeigen.DenseMatrix[] createPointCloud(float[] inverseDepth, int width, int height) {
		
		jeigen.DenseMatrix[] pointCloud = new jeigen.DenseMatrix[width*height];
		
		double fxInv = Constants.fxInv;
		double fyInv = Constants.fyInv;
		double cxInv = Constants.cxInv;
		double cyInv = Constants.cyInv;
		
		int pixelIndex = 0;
		for (int y=0 ; y<height ; y++) {
			for (int x=0 ; x<width ; x++) {
				
				float depth = inverseDepth[pixelIndex];
				// Skip if depth is not valid
				if(depth < 0) {
					continue;
				}
				//System.out.println(x + "," + y + ": " + depth);
				
				pointCloud[pixelIndex] = (new jeigen.DenseMatrix(
						new double[][]{{fxInv*x + cxInv},
									   {fyInv*y + cyInv},
									   {1}})).div(depth);
				
				//System.out.println(pointCloud[]);
				
				pixelIndex ++;
			}
		}
		
		
		return pointCloud;
	}
	
	public int width() {
		return frame.width();
	}
	
	public int height() {
		return frame.height();
	}
	
	public void writePointCloudToFile(String filename, jeigen.DenseMatrix[] pointCloud, int width, int height) throws FileNotFoundException, UnsupportedEncodingException {
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

	
}
