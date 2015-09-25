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
	
	/*
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
		
		///*
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
		//

		
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
	*/
	
	/**
	 * Create 3D points from inverse depth values
	 */
	public jeigen.DenseMatrix[] createPointCloud(float[] inverseDepth,
			float[] inverseDepthVariance, int width, int height, int level) {
		
		//System.out.println("createPointCloud: " + level);
		
		jeigen.DenseMatrix[] pointCloud = new jeigen.DenseMatrix[width*height];
		
		double fxInv = Constants.fxInv[level];
		double fyInv = Constants.fyInv[level];
		double cxInv = Constants.cxInv[level];
		double cyInv = Constants.cyInv[level];
		/*
		System.out.println("CreatePointCloud: L"+level + " " + fxInv + " "
															 + fyInv + " "
															 + cxInv + " "
															 + cyInv + " ");
		*/
		int pointsNum = 0;
		
		for (int x=1 ; x<width-1 ; x++) {
			for (int y=1 ; y<height-1 ; y++) {
				
				// Index to reference pixel
				int idx = x + y*width;
				
				// Get idepth, variance
				float idepth = inverseDepth[idx];
				float var = inverseDepthVariance[idx];

				//System.out.println("point1: " + idx + " - " + pointCloud[idx]);
				
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
				
				//System.out.println("point2: " + idx + " - " + pointCloud[idx]);
				
				pointsNum ++;
			}
		}
		System.out.println("MakePointCloud: " + level + " " + pointsNum);
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
	
	public static void writePointCloudToFile(String filename, TrackingReference ref) throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(filename, "ASCII");
	
		writer.println(3);
		
		float[] idepth = ref.keyframe.inverseDepthLvl[0];
		float[] idepth_var = ref.keyframe.inverseDepthVarianceLvl[0];
		
		int width = ref.width(0);
		int height = ref.height(0);
		
		int size = ref.width(0) * ref.height(0);
		
		double fxi = Constants.fxInv[0];
		double fyi = Constants.fyInv[0];
		double cxi = Constants.cxInv[0];
		double cyi = Constants.cyInv[0];
		
		for(int y=1;y<height-1;y++) {
			for(int x=1;x<width-1;x++){
				if(idepth[x+y*width] <= 0)
					continue;

				float depth = 1 / idepth[x+y*width];
				float depth4 = depth*depth; depth4*= depth4;


				//if(idepth_var[x+y*width] * depth4 > my_scaledTH)
				//	continue;

				//if(idepth_var[x+y*width] * depth4 * my_scale*my_scale > my_absTH)
				//	continue;

//				if(my_minNearSupport > 1)
//				{
//					int nearSupport = 0;
//					for(int dx=-1;dx<2;dx++)
//						for(int dy=-1;dy<2;dy++)
//						{
//							int idx = x+dx+(y+dy)*width;
//							if(originalInput[idx].idepth > 0)
//							{
//								float diff = originalInput[idx].idepth - 1.0f / depth;
//								if(diff*diff < 2*originalInput[x+y*width].idepth_var)
//									nearSupport++;
//							}
//						}
//
//					if(nearSupport < my_minNearSupport)
//						continue;
//				}

//				tmpBuffer[vertexBufferNumPoints].point[0] = (x*fxi + cxi) * depth;
//				tmpBuffer[vertexBufferNumPoints].point[1] = (y*fyi + cyi) * depth;
//				tmpBuffer[vertexBufferNumPoints].point[2] = depth;
//
//				tmpBuffer[vertexBufferNumPoints].color[3] = 100;
//				tmpBuffer[vertexBufferNumPoints].color[2] = originalInput[x+y*width].color[0];
//				tmpBuffer[vertexBufferNumPoints].color[1] = originalInput[x+y*width].color[1];
//				tmpBuffer[vertexBufferNumPoints].color[0] = originalInput[x+y*width].color[2];
//
//				vertexBufferNumPoints++;
//				displayed++;
				

				writer.printf("%.6f ",(x*fxi + cxi) * depth);
				writer.printf("%.6f ", (y*fyi + cyi) * depth);
				writer.printf("%.6f\n", depth);
				
			}
		}
//		for (int i=0 ; i<size ; i++) {
//			
//			writer.printf("%.6f ", pointCloud[i].get(0, 0));
//			writer.printf("%.6f ", pointCloud[i].get(1, 0));
//			writer.printf("%.6f\n", pointCloud[i].get(2, 0));
//			
// 		}
		
		writer.close();
	}

	public void importFrame(Frame currentKeyFrame) {
		
		keyframe = currentKeyFrame;
		
	}

	
}
