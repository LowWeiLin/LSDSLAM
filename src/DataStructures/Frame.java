package DataStructures;
import jeigen.DenseMatrix;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import DepthEstimation.DepthMapPixelHypothesis;
import LieAlgebra.SIM3;
import LieAlgebra.Vec;
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
	
	public Mat[] imageGradientMaxLvl;	// Gradient magnitude
	public float[][] imageGradientMaxArrayLvl; // Array of image gradient data for fast reading

	// Depth info
	public float[][] inverseDepthLvl;
	public float[][] inverseDepthVarianceLvl;
	public float meanIdepth = 0;
	public int numPoints = 0;
	public boolean hasIDepthBeenSet = false;
	public boolean depthHasBeenUpdatedFlag = false;
	
	
	// Pose
	public FramePoseStruct pose;
	
	//
	int id = 0;
	
	

	// Temporary values
	public int referenceID;
	public int referenceLevel;
	public double distSquared;
	public jeigen.DenseMatrix K_otherToThis_R; // 3x3 mat
	public jeigen.DenseMatrix K_otherToThis_t; // Vec3
	public jeigen.DenseMatrix otherToThis_t; // Vec3
	public jeigen.DenseMatrix K_thisToOther_t; // vec3
	public jeigen.DenseMatrix thisToOther_R; // 3x3 mat
	public jeigen.DenseMatrix otherToThis_R_row0; // vec3
	public jeigen.DenseMatrix otherToThis_R_row1; // vec3
	public jeigen.DenseMatrix otherToThis_R_row2; // vec3
	public jeigen.DenseMatrix thisToOther_t; // vec3
	
	
	
	public Frame(Mat image) {
		
		// Pose
		pose = new FramePoseStruct(this);
		
		// Initialize arrays
		this.imageLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageArrayLvl = new byte[Constants.PYRAMID_LEVELS][];
		this.imageGradientXLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientXArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientYLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientYArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientMaxLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientMaxArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.inverseDepthLvl = new float[Constants.PYRAMID_LEVELS][];
		this.inverseDepthVarianceLvl = new float[Constants.PYRAMID_LEVELS][];
		
		// Set level 0 image
		this.imageLvl[0] = image;
		this.imageArrayLvl[0] = new byte[(int) imageLvl[0].total()];
		this.imageLvl[0].get(0, 0, imageArrayLvl[0]);
		toSigned(imageArrayLvl[0]);
		
		// Set level 0 gradient
		this.imageGradientXLvl[0] = gradientX(imageLvl[0]);
		this.imageGradientYLvl[0] = gradientY(imageLvl[0]);
		// Max gradient
		this.imageGradientMaxLvl[0] = gradientMax(imageGradientXLvl[0], imageGradientYLvl[0]);
		
		
		this.imageGradientXArrayLvl[0] = new float[(int) imageGradientXLvl[0].total()];
		this.imageGradientXLvl[0].get(0, 0, imageGradientXArrayLvl[0]);
		this.imageGradientYArrayLvl[0] = new float[(int) imageGradientYLvl[0].total()];
		this.imageGradientYLvl[0].get(0, 0, imageGradientYArrayLvl[0]);
		this.imageGradientMaxArrayLvl[0] = new float[(int) imageGradientMaxLvl[0].total()];
		this.imageGradientMaxLvl[0].get(0, 0, imageGradientMaxArrayLvl[0]);
		
		// Generate lower levels
		for (int i=1 ; i<Constants.PYRAMID_LEVELS ; i++) {
			
			// Image
			this.imageLvl[i] = new Mat();
			Imgproc.pyrDown(this.imageLvl[i-1], this.imageLvl[i]);
			this.imageArrayLvl[i] = new byte[(int) imageLvl[i].total()];
			this.imageLvl[i].get(0, 0, imageArrayLvl[i]);
			toSigned(imageArrayLvl[i]);
			
			// Gradient
			this.imageGradientXLvl[i] = gradientX(imageLvl[i]);
			this.imageGradientYLvl[i] = gradientY(imageLvl[i]);
			this.imageGradientMaxLvl[i] = gradientMax(imageGradientXLvl[i], imageGradientYLvl[i]);

//			Imgproc.pyrDown(this.imageGradientXLvl[i-1], this.imageGradientXLvl[i]);
//			Imgproc.pyrDown(this.imageGradientYLvl[i-1], this.imageGradientYLvl[i]);
//			Imgproc.pyrDown(this.imageGradientMaxLvl[i-1], this.imageGradientMaxLvl[i]);
			
			
			// Highgui.imwrite("gradx-"+i+".jpg", this.imageGradientXLvl[i]);
			// Highgui.imwrite("gradY-"+i+".jpg", this.imageGradientYLvl[i]);
			
			this.imageGradientXArrayLvl[i] = new float[(int) imageGradientXLvl[i].total()];
			this.imageGradientXLvl[i].get(0, 0, imageGradientXArrayLvl[i]);
			this.imageGradientYArrayLvl[i] = new float[(int) imageGradientYLvl[i].total()];
			this.imageGradientYLvl[i].get(0, 0, imageGradientYArrayLvl[i]);
			this.imageGradientMaxArrayLvl[i] = new float[(int) imageGradientMaxLvl[i].total()];
			this.imageGradientMaxLvl[i].get(0, 0, imageGradientMaxArrayLvl[i]);
		}
	}
	
	Mat tempMat = new Mat();
	private Mat gradientMax(Mat gradX, Mat gradY) {
		Mat gradientMax = new Mat();
		
		Core.multiply(gradX, gradX, gradientMax);
		Core.multiply(gradY, gradY, tempMat);
		Core.add(gradientMax, tempMat, gradientMax);
		
		Core.sqrt(gradientMax, gradientMax);
		return gradientMax;
	}

	// Returns gradient of image
	public Mat gradientX(Mat image) {
		Mat gradientX = new Mat();
		Imgproc.Sobel(image, gradientX, CvType.CV_32F, 1, 0);
		
//		Mat kernel = new Mat(1, 3, CvType.CV_32F){
//							{
//							put(0,0,-0.5);
//							put(0,1,0);
//							put(0,2,0.5);
//							}
//						};
//         
//      Imgproc.filter2D(image, gradientX, CvType.CV_32F, kernel);
		
		return gradientX;
	}
	
	public Mat gradientY(Mat image) {
		Mat gradientY = new Mat();
		Imgproc.Sobel(image, gradientY, CvType.CV_32F, 0, 1);
		
//		Mat kernel = new Mat(3, 1, CvType.CV_32F){
//							{
//							put(0, 0,-0.5);
//							put(1, 0,0);
//							put(2, 0,0.5);
//							}
//						};
//         
//      Imgproc.filter2D(image, gradientY, CvType.CV_32F, kernel);
		
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

	public void setDepth(DepthMapPixelHypothesis[] newDepth) {
		

		int numIdepth = 0;
		float sumIdepth=0;
		
		int pixels = width(0)*height(0);
		for (int i=0 ; i<pixels ; i++) {
			if (newDepth[i].isValid && newDepth[i].idepth_smoothed >= -0.05) {
				inverseDepthLvl[0][i] = newDepth[i].idepth_smoothed;
				inverseDepthVarianceLvl[0][i] = newDepth[i].idepth_var_smoothed;
				numIdepth++;
				sumIdepth += newDepth[i].idepth_smoothed;
			} else {
				inverseDepthLvl[0][i] = -1;
				inverseDepthVarianceLvl[0][i] = -1;
			}
		}
		

		meanIdepth = sumIdepth / numIdepth;
		numPoints = numIdepth;
		
		// TODO: implement? needed?
//		idepthValid[0] = true;
//		idepthVarValid[0] = true;
		hasIDepthBeenSet = true;
		depthHasBeenUpdatedFlag = true;
		
	}
	
	void prepareForStereoWith(Frame other, SIM3 thisToOther, int level) {
		SIM3 otherToThis = thisToOther.inverse();

		K_otherToThis_R = Constants.K[0].mul(otherToThis.getRotationMat()).mul(otherToThis.getScale());
		otherToThis_t = Vec.array3ToVec(otherToThis.getTranslation());
		K_otherToThis_t = Constants.K[0].mul(otherToThis_t);



		thisToOther_t = Vec.array3ToVec(thisToOther.getTranslation());
		K_thisToOther_t = Constants.K[0].mul(thisToOther_t);
		thisToOther_R = thisToOther.getRotationMat().mul(thisToOther.getScale());
		otherToThis_R_row0 = thisToOther_R.col(0);
		otherToThis_R_row1 = thisToOther_R.col(1);
		otherToThis_R_row2 = thisToOther_R.col(2);

		distSquared = Vec.dot(otherToThis.getTranslation(), otherToThis.getTranslation());

		referenceID = other.id();
		referenceLevel = level;
	}
	
	
	public int id() {
		return id;
	}
	
}
