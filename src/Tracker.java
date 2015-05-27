import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import DataStructures.Frame;
import DataStructures.ReferenceFrame;
import Utils.Constants;



public class Tracker {
	
	
	/**
	 * Estimates the pose between a reference frame and a frame.
	 */
	void trackFrame(ReferenceFrame referenceFrame, Frame frame) {
		
		// TODO: Set an initial estimate
		
		// Get 3D points of reference frame
		referenceFrame.pointCloud = referenceFrame.createPointCloud(
				referenceFrame.inverseDepth,
				referenceFrame.width(), referenceFrame.height());
		
		// Test writing 3D point cloud
		try {
			referenceFrame.writePointCloudToFile("out.xyz", referenceFrame.pointCloud,
					referenceFrame.width(), referenceFrame.height());
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		
		// Get SO3 (?)
		// Get rotation, translation matrix
		jeigen.DenseMatrix rotationMat = null;
		jeigen.DenseMatrix translationVec = null;
		
		// Dummy values
		rotationMat = new jeigen.DenseMatrix(new double[][]{{1,0,0},{0,1,0},{0,0,1}});
		translationVec = new jeigen.DenseMatrix(new double[][]{{0},{0},{0}});
		

		// Calculate SSD
		float SSD = 0;
		int validPoints = 0;
		for (int i=0 ; i<referenceFrame.pointCloud.length ; i++) {
			// Each 3D point
			jeigen.DenseMatrix point = referenceFrame.pointCloud[i];
			
			// Warp to 2D image by estimate
			jeigen.DenseMatrix warpedPoint = point;//rotationMat.mmul(point).add(translationVec);
			
			// Image points
			double u = (warpedPoint.get(0, 0)/warpedPoint.get(2, 0))*Constants.fx + Constants.cx;
			double v = (warpedPoint.get(1, 0)/warpedPoint.get(2, 0))*Constants.fy + Constants.cy;
			
			// TODO: Check image points within bounds
			
			
			// TODO: Get interpolated value at image points, from frame, make this faster.
			// just get non-interpolated value for now
			float intensity = (int)frame.imageArray[(int)v*frame.width() + (int)u] & 0xFF;
			
			// convert signed/unsigned byte
			float referenceFrameIntensity = (int) referenceFrame.frame.imageArray[i] & 0xFF;
			
			// TODO: Change this
			float residual = intensity - referenceFrameIntensity;
			
			float squaredResidual = residual*residual;
			SSD += squaredResidual;
			validPoints++;
			
		}
		
		
		// Update estimate
		
		
		
	}
	
	
	byte[] array1 = null;
	byte[] array2 = null;
	// Calculate SSD between 2 Mats
	// image1 and image 2 must be of same size, 8bit, 1 channel
	int calcSSD(Mat image1, Mat image2) {
		
		int size = (int) (image1.total());
		
		// Reuse array of larger size, saves time creating a new array.
		if (array1 == null || array1.length < size) {
			array1 = new byte[size];
			array2 = new byte[size];
		}
		
		// Get image pixels
		image1.get(0, 0, array1);
		image2.get(0, 0, array2);
		
		// Calculate and return SSD
		int SSD = 0;
		int difference = 0;
		// TODO: multithread this loop?
		for (int i=0 ; i<size ; i++) {
			difference = array1[i] - array2[i];
			SSD += difference * difference;
		}
		return SSD;
	}
	
	
	// Entry point used to test Tracker
	public static void main(String[] args) {
		
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		
		// Read image
		Mat image1 = null;
		Mat image2 = null;
		image1 = Highgui.imread("test1.jpg");
		image2 = Highgui.imread("test1.jpg");
		
		// Convert to grayscale
		Imgproc.cvtColor(image1, image1, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(image2, image2, Imgproc.COLOR_RGB2GRAY);
		
		
		Frame frame1 = new Frame(image1);
		Frame frame2 = new Frame(image2);
		ReferenceFrame refFrame = new ReferenceFrame(frame1);
		
		
		// Track frame
		Tracker tracker = new Tracker();
		tracker.trackFrame(refFrame, frame2);
		
		
	}
	
}
