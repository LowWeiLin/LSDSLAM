import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import DataStructures.Frame;
import DataStructures.ReferenceFrame;
import LieAlgebra.SE3;
import LieAlgebra.Vec;
import Utils.Constants;



public class Tracker {
	
	
	/**
	 * Estimates the pose between a reference frame and a frame.
	 */
	void trackFrame(ReferenceFrame referenceFrame, Frame frame) {
		
		// Perform twiddle
		
		// Set an initial estimate
		double[] estimateVec6 = {0,0,0,0,0,0};
		SE3 frameToRefEstimate = SE3.exp(estimateVec6);

		double[] incrementVec6 = {0.1,0.1,0.1,0.05,0.05,0.05};
		double successMultiplier = 1.3;
		double failureMultiplier = 0.7;

		int iterationCount = 0;
		for (int level=Constants.SE3TRACKING_MAX_LEVEL-1 ;
				level>=Constants.SE3TRACKING_MIN_LEVEL ;
				level-=1) {
			
			incrementVec6 = new double[]{0.1,0.1,0.1,0.05,0.05,0.05};
			
			double minSSD = calculateSSD(referenceFrame, frame, frameToRefEstimate, level);
			System.out.println("Min SSD: " + minSSD);
			
			
			//while(true) {
			for (int iterationLevelCount=0 ; iterationLevelCount<70 ; iterationLevelCount++) {
				iterationCount++;
				double incrementMagnitude = Vec.magnitude(incrementVec6);
				
	
				System.out.println(iterationCount);
				System.out.println(incrementMagnitude);
				System.out.println("Min SSD: " + minSSD);
				
				System.out.println("IncrementVec: " + Arrays.toString(incrementVec6));
				System.out.println("Vec6: " + Arrays.toString(SE3.ln(frameToRefEstimate)));
				//System.out.println("Rotation: " + estimate.getRotationMat());
				//System.out.println("Translation: " + estimate.getTranslationMat());
				
				if (incrementMagnitude < 1e-6 || minSSD < 100) {
					break;
				}
				
				for (int i=0 ; i<incrementVec6.length ; i++) {
					
					estimateVec6[i] += incrementVec6[i];
					frameToRefEstimate = SE3.exp(estimateVec6);
					
					double SSD = calculateSSD(referenceFrame, frame, frameToRefEstimate, level);
					if (SSD < minSSD) {
						minSSD = SSD;
						incrementVec6[i] *= successMultiplier;
					} else {
						estimateVec6[i] -= 2*incrementVec6[i];
						frameToRefEstimate = SE3.exp(estimateVec6);
						SSD = calculateSSD(referenceFrame, frame, frameToRefEstimate, level);
						if (SSD < minSSD) {
							minSSD = SSD;
							incrementVec6[i] *= -successMultiplier;
						} else {
							estimateVec6[i] += incrementVec6[i];
							frameToRefEstimate = SE3.exp(estimateVec6);
							incrementVec6[i] *= failureMultiplier;
						}
					}
					
					
				}
	
			}
		}
		

		// Test writing 3D point cloud
		try {
			referenceFrame.writePointCloudToFile("out1.xyz", referenceFrame.pointCloudLvl[0],
					referenceFrame.width(0), referenceFrame.height(0));
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		

		jeigen.DenseMatrix rotationMat = frameToRefEstimate.getRotationMat();
		jeigen.DenseMatrix translationVec = frameToRefEstimate.getTranslationMat();
		for (int i=0 ; i<referenceFrame.pointCloudLvl[0].length ; i++) {
			// Each 3D point
			jeigen.DenseMatrix point = referenceFrame.pointCloudLvl[0][i];
			
			// Warp to 2D image by estimate
			jeigen.DenseMatrix warpedPoint = rotationMat.mmul(point).add(translationVec);
			referenceFrame.pointCloudLvl[0][i] = warpedPoint;
		}
		
		// Test writing 3D point cloud
		try {
			referenceFrame.writePointCloudToFile("out2.xyz", referenceFrame.pointCloudLvl[0],
					referenceFrame.width(0), referenceFrame.height(0));
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		
		System.out.println("IncrementVec: " + Arrays.toString(incrementVec6));
		System.out.println("Vec6: " + Arrays.toString(SE3.ln(frameToRefEstimate)));
		System.out.println("Rotation: " + frameToRefEstimate.getRotationMat());
		System.out.println("Translation: " + frameToRefEstimate.getTranslationMat());
		
		
	}
	
	int count = 0;
	public double calculateSSD(ReferenceFrame referenceFrame,
			Frame frame, SE3 frameToRefPose, int level) {
		count++;
		
		double fx = Constants.fx[level];
		double fy = Constants.fy[level];
		double cx = Constants.cx[level];
		double cy = Constants.cy[level];

		// Get 3D points of reference frame
		if (referenceFrame.pointCloudLvl[level] == null) {
			referenceFrame.pointCloudLvl[level] = referenceFrame.createPointCloud(
					referenceFrame.inverseDepthLvl[level],
					referenceFrame.width(level), referenceFrame.height(level), level);
		}
		
		// Get rotation, translation matrix
		jeigen.DenseMatrix rotationMat = frameToRefPose.getRotationMat();
		jeigen.DenseMatrix translationVec = frameToRefPose.getTranslationMat();
		
//		Mat debugImage = new Mat(referenceFrame.height(level), referenceFrame.width(level), CvType.CV_8UC1);
//		byte[] debugArray = new byte[(int) debugImage.total()];
//		debugImage.get(0, 0, debugArray);
		
		
		// Calculate SSD
		double SSD = 0;
		int validPoints = 0;
		for (int i=0 ; i<referenceFrame.pointCloudLvl[level].length ; i++) {
			// Each 3D point
			jeigen.DenseMatrix point = referenceFrame.pointCloudLvl[level][i];
			
			// Warp to 2D image by estimate
			jeigen.DenseMatrix warpedPoint = rotationMat.mmul(point).add(translationVec);
			
			// Image points
			double u = (warpedPoint.get(0, 0)/warpedPoint.get(2, 0))*fx + cx;
			double v = (warpedPoint.get(1, 0)/warpedPoint.get(2, 0))*fy + cy;
			
			// Check image points within bounds
			if (!(u>1 && v>1 && u<frame.width(level)-1 && v<frame.height(level)-1)) {
				continue;
			}
			
			// TODO: Get interpolated value at image points from frame.
			// just get non-interpolated value for now
			//float intensity = (int)frame.imageArray[(int)v*frame.width() + (int)u] & 0xFF;
			float intensity = interpolatedPixel(frame.imageArrayLvl[level], u, v, frame.width(level));

			//debugArray[i] = (byte)intensity;
			
			// convert signed/unsigned byte
			float referenceFrameIntensity = (int) referenceFrame.frame.imageArrayLvl[level][i] & 0xFF;
			
			// TODO: Change this according to paper, code.
			float residual = referenceFrameIntensity - intensity;
			
			float squaredResidual = residual*residual;
			SSD += squaredResidual;
			validPoints++;
			
		}
		
		//debugImage.put(0, 0, debugArray);
		//Highgui.imwrite("debugImage"+count+"-"+ SSD + "-" + validPoints + ".jpg", debugImage);
		
		// Some condition to make sure there are some valid points.
		if (validPoints <= frame.width(level)*frame.height(level)*0.5) {
			return Double.MAX_VALUE;
		}
		
		double ratio = SSD/validPoints;
		
//		System.out.println("SSD: " + SSD);
//		System.out.println("Valid points: " + validPoints);
//		System.out.println("Ratio: " + ratio);
		
		return SSD;
	}
	
	
	// Entry point used to test Tracker
	public static void main(String[] args) {
		
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		Constants.setK(500, 500, 350, 240);
		
		
		// Read image
		Mat image1 = null;
		Mat image2 = null;
		image1 = Highgui.imread("test1.jpg");
		image2 = Highgui.imread("test2.jpg");
		
		// Convert to grayscale
		Imgproc.cvtColor(image1, image1, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(image2, image2, Imgproc.COLOR_RGB2GRAY);
		
		// Reduce resolution
//		Imgproc.pyrDown(image1, image1);
//		Imgproc.pyrDown(image2, image2);
//		
//		Imgproc.pyrUp(image1, image1);
//		Imgproc.pyrUp(image2, image2);
//		
//		Imgproc.pyrDown(image1, image1);
//		Imgproc.pyrDown(image2, image2);
//		
//		Imgproc.pyrUp(image1, image1);
//		Imgproc.pyrUp(image2, image2);
		
		
		
//		Imgproc.pyrDown(image1, image1);
//		Imgproc.pyrDown(image2, image2);
//		Imgproc.pyrDown(image1, image1);
//		Imgproc.pyrDown(image2, image2);
//		Imgproc.pyrDown(image1, image1);
//		Imgproc.pyrDown(image2, image2);
		
		System.out.println("Image sizes: " +  image1.width() + ", " + image1.height());
		
		
		Frame frame1 = new Frame(image1);
		Frame frame2 = new Frame(image2);
		ReferenceFrame refFrame = new ReferenceFrame(frame1);
		
		
		// Track frame
		Tracker tracker = new Tracker();
		tracker.trackFrame(refFrame, frame2);
		
	}
	
	
	static float interpolatedPixel(byte[] dataArray, double x, double y, int width) {

		int ix = (int)x;
		int iy = (int)y;
		float dx = (float) (x - ix);
		float dy = (float) (y - iy);
		float dxdy = dx*dy;
		int bp = ix+iy*width;
		
		float res =   dxdy 			* (float)((int)dataArray[bp+1+width] & 0xFF)
					+ (dy-dxdy) 	* (float)((int)dataArray[bp+width] & 0xFF)
					+ (dx-dxdy) 	* (float)((int)dataArray[bp+1] & 0xFF)
					+ (1-dx-dy+dxdy)* (float)((int)dataArray[bp] & 0xFF);

		return res;
		
	}
	
}
