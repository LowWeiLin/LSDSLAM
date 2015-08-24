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
	public float[][] imageArrayLvl; // Array of image data for fast reading

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
	public boolean hasIDepthBeenSet = false;
	public boolean depthHasBeenUpdatedFlag = false;
	
	
	// Graph values
	static int totalFrames = 0;
	int id;
	public FramePoseStruct pose;
	
	

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
	

	// statistics
	public float initialTrackedResidual;
	public int numFramesTrackedOnThis;
	public int numMappedOnThis;
	public int numMappedOnThisTotal;
	public float meanIdepth;
	public int numPoints;
	public int idxInKeyframes;
	public float edgeErrorSum, edgesNum;
	public int numMappablePixels;
	public float meanInformation;
	
	
	public Frame(Mat image) {
		
		// Set frame id
		id = totalFrames;
		totalFrames++;
		
		
		// Pose
		pose = new FramePoseStruct(this);
		
		// Initialize arrays
		this.imageLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientXLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientXArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientYLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientYArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.imageGradientMaxLvl = new Mat[Constants.PYRAMID_LEVELS];
		this.imageGradientMaxArrayLvl = new float[Constants.PYRAMID_LEVELS][];
		this.inverseDepthLvl = new float[Constants.PYRAMID_LEVELS][];
		this.inverseDepthVarianceLvl = new float[Constants.PYRAMID_LEVELS][];
		
		

		// Convert image to float type
		image.convertTo(image, CvType.CV_32F);
		
		// Set level 0 image
		this.imageLvl[0] = image;
		this.imageArrayLvl[0] = new float[(int) imageLvl[0].total()];
		this.imageLvl[0].get(0, 0, imageArrayLvl[0]);
		toSigned(imageArrayLvl[0]);
		
		// Set level 0 gradient
		this.imageGradientXArrayLvl[0] = gradientX(imageArrayLvl[0], 0);
		this.imageGradientYArrayLvl[0] = gradientY(imageArrayLvl[0], 0);
		
//		this.imageGradientXLvl[0] = new Mat();
//		this.imageGradientYLvl[0] = new Mat();
//		this.imageGradientXLvl[0].put(0, 0, imageGradientXArrayLvl[0]);
//		this.imageGradientYLvl[0].put(0, 0, imageGradientYArrayLvl[0]);
		
		
		// Max gradient
		this.imageGradientMaxLvl[0] = gradientMax(imageGradientXLvl[0], imageGradientYLvl[0]);
		
		this.imageGradientMaxArrayLvl[0] = new float[(int) imageGradientMaxLvl[0].total()];
		this.imageGradientMaxLvl[0].get(0, 0, imageGradientMaxArrayLvl[0]);
		
		// Generate lower levels
		for (int i=1 ; i<Constants.PYRAMID_LEVELS ; i++) {
			
			// Image
			this.imageLvl[i] = new Mat();
			Imgproc.pyrDown(this.imageLvl[i-1], this.imageLvl[i]);
			this.imageArrayLvl[i] = new float[(int) imageLvl[i].total()];
			this.imageLvl[i].get(0, 0, imageArrayLvl[i]);
			toSigned(imageArrayLvl[i]);
			
			// Gradient
			this.imageGradientXArrayLvl[i] = gradientX(imageArrayLvl[i], i);
			this.imageGradientYArrayLvl[i] = gradientY(imageArrayLvl[i], i);

//			this.imageGradientXLvl[i] = new Mat();
//			this.imageGradientYLvl[i] = new Mat();
//			this.imageGradientXLvl[i].put(0, 0, imageGradientXArrayLvl[i]);
//			this.imageGradientYLvl[i].put(0, 0, imageGradientYArrayLvl[i]);
			
			// Max gradient
			
			this.imageGradientMaxLvl[i] = gradientMax(imageGradientXLvl[i], imageGradientYLvl[i]);
			
			this.imageGradientMaxArrayLvl[i] = new float[(int) imageGradientMaxLvl[i].total()];
			this.imageGradientMaxLvl[i].get(0, 0, imageGradientMaxArrayLvl[i]);
			
//			Imgproc.pyrDown(this.imageGradientXLvl[i-1], this.imageGradientXLvl[i]);
//			Imgproc.pyrDown(this.imageGradientYLvl[i-1], this.imageGradientYLvl[i]);
//			Imgproc.pyrDown(this.imageGradientMaxLvl[i-1], this.imageGradientMaxLvl[i]);
			
			
			//Highgui.imwrite("gradX-"+this.id+"-"+i+".jpg", this.imageGradientXLvl[i]);
			//Highgui.imwrite("gradY-"+this.id+"-"+i+".jpg", this.imageGradientYLvl[i]);
			//Highgui.imwrite("gradmax-"+this.id+"-"+i+".jpg", this.imageGradientMaxLvl[i]);
			
			
			
			
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
	public float[] gradientX(float[] imageArrayLvl, int level) {
		float[] imageGradientXArray = new float[imageArrayLvl.length];
		int w = width(level);
		int h = height(level);
		
		for (int i=w ; i<=w*(h-1) ; i++) {
			imageGradientXArray[i] =  0.5f*(imageArrayLvl[i-1] - imageArrayLvl[i+1]);
		}
		
		return imageGradientXArray;
	}
	
	public float[] gradientY(float[] imageArrayLvl, int level) {
		float[] imageGradientYArray = new float[imageArrayLvl.length];
		int w = width(level);
		int h = height(level);

		System.out.println(w + " "+ h);
		
		for (int i=w ; i<w*(h-1) ; i++) {
			imageGradientYArray[i] =  0.5f*(imageArrayLvl[i+w] - imageArrayLvl[i-w]);
		}
		
		return imageGradientYArray;
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
	public static void toSigned(float[] imageArrayLvl2) {
		for (int i=0 ; i<imageArrayLvl2.length ; i++) {
			//imageArrayLvl2[i] &= 0xFF;
			//TODO: 
		}
	}

	public void setDepth(DepthMapPixelHypothesis[] newDepth) {
		

		int numIdepth = 0;
		float sumIdepth=0;
		
		int pixels = width(0)*height(0);
		if (inverseDepthLvl[0] == null) {
			inverseDepthLvl[0] = new float[pixels];
		}
		if (inverseDepthVarianceLvl[0] == null) {
			inverseDepthVarianceLvl[0] = new float[pixels];
		}
		
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
		
		
		// Do lower levels
		
		for (int level=1 ; level<Constants.PYRAMID_LEVELS ; level++) {
			int pixelsLvl = width(level)*height(level);
			// Initialize arrays
			if (inverseDepthLvl[level] == null) {
				inverseDepthLvl[level] = new float[pixelsLvl];
			}
			if (inverseDepthVarianceLvl[level] == null) {
				inverseDepthVarianceLvl[level] = new float[pixelsLvl];
			}
			

			float[] idepthSource = inverseDepthLvl[level - 1];
			float[] idepthVarSource = inverseDepthVarianceLvl[level - 1];
			float[] idepthDest = inverseDepthLvl[level];
			float[] idepthVarDest = inverseDepthVarianceLvl[level];
			
			int width = width(level);
			int height = height(level);
			int sw = width(level - 1);
			
			for(int y=0;y<height;y++) {
				for(int x=0;x<width;x++) {
					int idx = 2*(x+y*sw);
					int idxDest = (x+y*width);

					float idepthSumsSum = 0;
					float ivarSumsSum = 0;
					int num=0;

					// build sums
					float ivar;
					float var = idepthVarSource[idx];
					if(var > 0)
					{
						ivar = 1.0f / var;
						ivarSumsSum += ivar;
						idepthSumsSum += ivar * idepthSource[idx];
						num++;
					}

					var = idepthVarSource[idx+1];
					if(var > 0)
					{
						ivar = 1.0f / var;
						ivarSumsSum += ivar;
						idepthSumsSum += ivar * idepthSource[idx+1];
						num++;
					}

					var = idepthVarSource[idx+sw];
					if(var > 0)
					{
						ivar = 1.0f / var;
						ivarSumsSum += ivar;
						idepthSumsSum += ivar * idepthSource[idx+sw];
						num++;
					}

					var = idepthVarSource[idx+sw+1];
					if(var > 0)
					{
						ivar = 1.0f / var;
						ivarSumsSum += ivar;
						idepthSumsSum += ivar * idepthSource[idx+sw+1];
						num++;
					}
					
					if(num > 0)
					{
						float depth = ivarSumsSum / idepthSumsSum;
						idepthDest[idxDest] = 1.0f / depth;
						idepthVarDest[idxDest] = num / ivarSumsSum;
					}
					else
					{
						idepthDest[idxDest] = -1;
						idepthVarDest[idxDest] = -1;
					}
				}
			}

			//idepthValid[level] = true;
			//idepthVarValid[level] = true;
			
		}
		
	}
	
	public void prepareForStereoWith(Frame other, SIM3 thisToOther, int level) {
		SIM3 otherToThis = thisToOther.inverse();

		K_otherToThis_R = Constants.K[0].mmul(otherToThis.getRotationMat()).mul(otherToThis.getScale());
		otherToThis_t = Vec.array3ToVec(otherToThis.getTranslation());
		K_otherToThis_t = Constants.K[0].mmul(otherToThis_t);



		thisToOther_t = Vec.array3ToVec(thisToOther.getTranslation());
		K_thisToOther_t = Constants.K[0].mmul(thisToOther_t);
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
	
	public SIM3 getScaledCamToWorld() {
		return pose.getCamToWorld();
	}
	
	public boolean hasTrackingParent() {
		return pose.trackingParent != null;
	}
	
	public Frame getTrackingParent() {
		return pose.trackingParent.frame;
	}
	
}
