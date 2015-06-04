package DataStructures;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;


public class Frame {

	// Gray scale image
	public Mat[] imageLvl;
	public byte[][] imageArrayLvl; // Array of image data for fast reading

	// Gradient
	public Mat[] imageGradientXLvl;
	public float[][] imageGradientXArrayLvl; // Array of image gradient data for fast reading
	public Mat[] imageGradientYLvl;
	public float[][] imageGradientYArrayLvl; // Array of image gradient data for fast reading

	
	public Frame(Mat image) {

		// Initialize arrays
		this.imageLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageArrayLvl = new byte[Constants.PYRAMID_LEVELS][];
		this.imageGradientXLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientXArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientYLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientYArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		
		
		// Set level 0 image
		this.imageLvl[0] = image;
		this.imageArrayLvl[0] = new byte[(int) imageLvl[0].total()];
		this.imageLvl[0].get(0, 0, imageArrayLvl[0]);
		toSigned(imageArrayLvl[0]);
		
		// Set level 0 gradient
		this.imageGradientXLvl[0] = gradientX(imageLvl[0]);
		this.imageGradientYLvl[0] = gradientY(imageLvl[0]);
		
		this.imageGradientXArrayLvl[0] = new float[(int) imageGradientXLvl[0].total()];
		this.imageGradientXLvl[0].get(0, 0, imageGradientXArrayLvl[0]);
		this.imageGradientYArrayLvl[0] = new float[(int) imageGradientYLvl[0].total()];
		this.imageGradientYLvl[0].get(0, 0, imageGradientYArrayLvl[0]);
		
		// Generate lower levels
		for (int i=1 ; i<Constants.PYRAMID_LEVELS ; i++) {
			
			// Image
			this.imageLvl[i] = new Mat();
			Imgproc.pyrDown(this.imageLvl[i-1], this.imageLvl[i]);
			this.imageArrayLvl[i] = new byte[(int) imageLvl[i].total()];
			this.imageLvl[i].get(0, 0, imageArrayLvl[i]);
			toSigned(imageArrayLvl[i]);
			
			// Gradient
			this.imageGradientXLvl[i] = new Mat();
			this.imageGradientYLvl[i] = new Mat();
			Imgproc.pyrDown(this.imageGradientXLvl[i-1], this.imageGradientXLvl[i]);
			Imgproc.pyrDown(this.imageGradientYLvl[i-1], this.imageGradientYLvl[i]);
			
			// Highgui.imwrite("gradx-"+i+".jpg", this.imageGradientXLvl[i]);
			// Highgui.imwrite("gradY-"+i+".jpg", this.imageGradientYLvl[i]);
			
			this.imageGradientXArrayLvl[i] = new float[(int) imageGradientXLvl[i].total()];
			this.imageGradientXLvl[i].get(0, 0, imageGradientXArrayLvl[i]);
			this.imageGradientYArrayLvl[i] = new float[(int) imageGradientYLvl[i].total()];
			this.imageGradientYLvl[i].get(0, 0, imageGradientYArrayLvl[i]);
		}
	}
	
	// Returns gradient of image
	public Mat gradientX(Mat image) {
		Mat gradientX = new Mat();
		Imgproc.Sobel(image, gradientX, CvType.CV_32F, 1, 0);
		return gradientX;
	}
	
	public Mat gradientY(Mat image) {
		Mat gradientY = new Mat();
		Imgproc.Sobel(image, gradientY, CvType.CV_32F, 1, 0);
		return gradientY;
	}
	
//	public int width() {
//		return imageLvl[0].width();
//	}
//	
//	public int height() {
//		return imageLvl[0].height();
//	}
	
	public int width(int level) {
		return imageLvl[level].width();
	}
	
	public int height(int level) {
		return imageLvl[level].height();
	}
	

	// Applies & 0xFF to each element, to convert from unsigned to signed values.
	public static void toSigned(byte[] byteArray) {
		for (int i=0 ; i<byteArray.length ; i++) {
			byteArray[i] &= 0xFF;
		}
	}
	
}
