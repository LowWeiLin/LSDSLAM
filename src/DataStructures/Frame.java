package DataStructures;
import java.util.Arrays;

import jeigen.DenseMatrix;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
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
	
	// Tracking
	public boolean[] refPixelWasGood;
	
	
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
		//toSigned(imageArrayLvl[0]);
		
		float sum = 0;
		for (float f : imageArrayLvl[0])
			sum+=f;
		
		System.out.println("SUM: " + sum);
		
		// Set level 0 gradient
		this.imageGradientXArrayLvl[0] = gradientX(imageArrayLvl[0], 0);
		this.imageGradientYArrayLvl[0] = gradientY(imageArrayLvl[0], 0);
		
//		this.imageGradientXLvl[0] = new Mat();
//		this.imageGradientYLvl[0] = new Mat();
//		this.imageGradientXLvl[0].put(0, 0, imageGradientXArrayLvl[0]);
//		this.imageGradientYLvl[0].put(0, 0, imageGradientYArrayLvl[0]);
		
		
		// Max gradient
		this.imageGradientMaxArrayLvl[0] = gradientMax(0);
		
		// Generate lower levels
		for (int i=1 ; i<Constants.PYRAMID_LEVELS ; i++) {
			
			// Image

			imageArrayLvl[i] = new float[imageArrayLvl[i-1].length/4];
			buildImageLevel(imageArrayLvl[i-1], imageArrayLvl[i], i);
			this.imageLvl[i] = new Mat(new Size(width(i-1)/2, height(i-1)/2), CvType.CV_32F);
			this.imageLvl[i].put(0,0,imageArrayLvl[i]);
			
			//this.imageLvl[i] = new Mat();
			//Imgproc.pyrDown(this.imageLvl[i-1], this.imageLvl[i]);
			
			//this.imageArrayLvl[i] = new float[(int) imageLvl[i].total()];
			//this.imageLvl[i].get(0, 0, imageArrayLvl[i]);
			//toSigned(imageArrayLvl[i]);
			
			// Gradient
			this.imageGradientXArrayLvl[i] = gradientX(imageArrayLvl[i], i);
			this.imageGradientYArrayLvl[i] = gradientY(imageArrayLvl[i], i);
			this.imageGradientXLvl[i] = new Mat(new Size(width(i), height(i)), CvType.CV_32F);
			this.imageGradientXLvl[i].put(0,0,imageGradientXArrayLvl[i]);
			this.imageGradientYLvl[i] = new Mat(new Size(width(i), height(i)), CvType.CV_32F);
			this.imageGradientYLvl[i].put(0,0,imageGradientYArrayLvl[i]);

			// Max gradient
			this.imageGradientMaxArrayLvl[i] = gradientMax(i);
			this.imageGradientMaxLvl[i] = new Mat(new Size(width(i), height(i)), CvType.CV_32F);
			this.imageGradientMaxLvl[i].put(0,0,imageGradientMaxArrayLvl[i]);
			
//			Highgui.imwrite("img-"+this.id+"-"+i+".jpg", this.imageLvl[i]);
//			Highgui.imwrite("gradX-"+this.id+"-"+i+".jpg", this.imageGradientXLvl[i]);
//			Highgui.imwrite("gradY-"+this.id+"-"+i+".jpg", this.imageGradientYLvl[i]);
//			Highgui.imwrite("gradmax-"+this.id+"-"+i+".jpg", this.imageGradientMaxLvl[i]);
			
			
		}
	}
	
	private float[] gradientMax(int level) {
		
		int w = width(level);
		int h = height(level);
		
		float[] gradientMax = new float[w * h];
		float[] gradientMaxTemp = new float[w * h];
		
		// 1. write abs gradients in real data.
		for(int i=w ; i<w*(h-1) ; i++) {
			float dx = imageGradientXArrayLvl[level][i];
			float dy = imageGradientYArrayLvl[level][i];
			gradientMax[i] = (float) Math.sqrt(dx*dx + dy*dy);
		}
		
		// Steps 2 and 3 for setting pixel to largest gradient in 3x3 neighborhood
		
		// 2. smear up/down direction into temp buffer
		for (int i=w+1 ; i<w*(h-1)-1 ; i++) {
			float g1 = gradientMax[i-w]; // gradient of pixel on top
			float g2 = gradientMax[i];	 // gradient of current pixel
			if(g1 < g2) {
				g1 = g2;
			}
			float g3 = gradientMax[i+w]; // gradient of pixel below
			
			// Set current gradient to largest gradient of top/current/bottom pixel
			if(g1 < g3) {
				gradientMaxTemp[i] = g3;
			} else {
				gradientMaxTemp[i] = g1;
			}
		}
		
		int numMappablePixels = 0;
		// 2. smear left/right direction into real data
		for (int i=w+1 ; i<w*(h-1)-1 ; i++) {
			float g1 = gradientMaxTemp[i-1]; // Pixel on left
			float g2 = gradientMaxTemp[i];	 // Current pixel
			if(g1 < g2) {
				g1 = g2;
			}
			float g3 = gradientMaxTemp[i+1]; // Pixel on right
			// Set to largest gradient
			if(g1 < g3) {
				gradientMax[i] = g3;
				if(g3 >= Constants.MIN_ABS_GRAD_CREATE)
					numMappablePixels++;
			} else {
				gradientMax[i] = g1;
				if(g1 >= Constants.MIN_ABS_GRAD_CREATE)
					numMappablePixels++;
			}
		}

		if(level==0)
			this.numMappablePixels = numMappablePixels;
		
		
		return gradientMax;
	}

	// Returns gradient of image
	public float[] gradientX(float[] imageArrayLvl, int level) {
		float[] imageGradientXArray = new float[imageArrayLvl.length];
		int w = width(level);
		int h = height(level);
		
		for (int i=w ; i<=w*(h-1) ; i++) {
			imageGradientXArray[i] =  0.5f*(imageArrayLvl[i+1] - imageArrayLvl[i-1]);
		}
		
		return imageGradientXArray;
	}
	
	public float[] gradientY(float[] imageArrayLvl, int level) {
		float[] imageGradientYArray = new float[imageArrayLvl.length];
		int w = width(level);
		int h = height(level);
		
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
//	public static void toSigned(float[] imageArrayLvl2) {
//		for (int i=0 ; i<imageArrayLvl2.length ; i++) {
//			//imageArrayLvl2[i] &= 0xFF;
//			//TODO: 
//		}
//	}

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
			buildIDepthAndIDepthVar(level);
		}
		
	}
	
	public void buildIDepthAndIDepthVar(int level) {
		if (level <= 0 ) {
			System.err.println("buildIDepthAndIDepthVar: Invalid level parameter!");
			return;
		}
		//System.out.println("buildIDepthAndIDepthVar: building level " + level);
		
		int width = width(level);
		int height = height(level);
		
		int sw = width(level - 1);
		
		inverseDepthLvl[level] = new float[width*height];
		inverseDepthVarianceLvl[level] = new float[width*height];
		
		
		float[] idepthSource = inverseDepthLvl[level - 1];
		float[] idepthVarSource = inverseDepthVarianceLvl[level - 1];

		float[] idepthDest = inverseDepthLvl[level];
		float[] idepthVarDest = inverseDepthVarianceLvl[level];

		
		for(int y=0 ; y<height ; y++) {
			for(int x=0 ; x<width ; x++) {
				int idx = 2*(x+y*sw);		// Index for level - 1
				int idxDest = (x+y*width);	// Index for level
				
				// Sums 4 pixels
				float idepthSumsSum = 0;
				float ivarSumsSum = 0;
				int num=0;
				
				// build sums
				float ivar;
				float var = idepthVarSource[idx];
				if(var > 0) {
					ivar = 1.0f / var;
					ivarSumsSum += ivar;
					idepthSumsSum += ivar * idepthSource[idx];
					num++;
				}

				var = idepthVarSource[idx+1];
				if(var > 0) {
					ivar = 1.0f / var;
					ivarSumsSum += ivar;
					idepthSumsSum += ivar * idepthSource[idx+1];
					num++;
				}

				var = idepthVarSource[idx+sw];
				if(var > 0) {
					ivar = 1.0f / var;
					ivarSumsSum += ivar;
					idepthSumsSum += ivar * idepthSource[idx+sw];
					num++;
				}

				var = idepthVarSource[idx+sw+1];
				if(var > 0) {
					ivar = 1.0f / var;
					ivarSumsSum += ivar;
					idepthSumsSum += ivar * idepthSource[idx+sw+1];
					num++;
				}
				
				if(num > 0) {
					float depth = ivarSumsSum / idepthSumsSum;
					idepthDest[idxDest] = 1.0f / depth;
					idepthVarDest[idxDest] = num / ivarSumsSum;
				} else {
					idepthDest[idxDest] = -1;
					idepthVarDest[idxDest] = -1;
				}
			}
		}
		
		
	}
	
	public void buildImageLevel(float[] imageArraySrc, float[] imageArrayDst, int level) {

		int width = width(level - 1);
		int height = height(level - 1);
		float[] source = imageArraySrc;
		float[] dest = imageArrayDst;
		

		int wh = width*height;
		int srcIdx = 0;
		int dstIdx = 0;
		for(int y=0;y<wh;y+=width*2) {
			for(int x=0;x<width;x+=2) {
				srcIdx = x + y;
				dest[dstIdx] = (source[srcIdx] +
						source[srcIdx+1] +
						source[srcIdx+width] +
						source[srcIdx+1+width]) * 0.25f;
				dstIdx++;
			}
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
		
		//System.out.println("distSquared " + distSquared);
		
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
	
	public boolean[] refPixelWasGoodNoCreate() {
		return refPixelWasGood;
	}

	public boolean[] refPixelWasGood() {
		if (refPixelWasGood == null) {
			int width = width(Constants.SE3TRACKING_MIN_LEVEL);
			int height = height(Constants.SE3TRACKING_MIN_LEVEL);
			refPixelWasGood = new boolean[width*height];
			Arrays.fill(refPixelWasGood, true);
		}
		return refPixelWasGood;
	}

	public void clear_refPixelWasGood() {
		refPixelWasGood = null;
	}
}
