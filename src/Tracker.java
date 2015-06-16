import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

import jeigen.DenseMatrix;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import DataStructures.Frame;
import DataStructures.TrackingReference;
import LieAlgebra.SE3;
import LieAlgebra.Vec;
import Utils.Constants;



public class Tracker {
	
	// Settings variables
	public static final int maxIterations[] = {5, 20, 50, 100, 100, 100};
	public static final int maxItsPerLvl[] = {5, 20, 50, 100, 100, 100};
	public static float[] lambdaInitial = new float[Constants.PYRAMID_LEVELS];
	public static float[] convergenceEps = new float[Constants.PYRAMID_LEVELS];
	public static final float varWeight = 1.0f;
	public static final float huberD = 3.0f;
	public static final float cameraPixelNoise2 = 4*4;
	
	public static final float lambdaSuccessFac = 0.5f;
	public static final float lambdaFailFac = 2.0f;
	public static final float stepSizeMin[] = {1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f};
	
	
	// Variables set when tracking
	int warpedCount = 0; // Number of pixels warped into new image bounds
	
	
	// Buffers for holding data of pixels warped into new image bounds
	// Set in calculateResidualAndBuffers()
	// Maximum size of buffers is width(0)*height(0)
	// Current size is warpedCount
	boolean initialized = false;
	float[] bufWarpedResidual;
	float[] bufWarpedDx;
	float[] bufWarpedDy;
	float[] bufWarpedX;
	float[] bufWarpedY;
	float[] bufWarpedZ;
	float[] bufInvDepth;
	float[] bufInvDepthVariance;
	float[] bufWeightP;
	
	
	
	public Tracker() {
		
		// Set lambdaInitial values to 0
		Arrays.fill(lambdaInitial, 0);
		
		// Set convergence epsilon
		Arrays.fill(convergenceEps, 0.999f);
		
	}
	
	
	public void initialize(int width, int height) {
		// Only initialize once
		if (!this.initialized) {
			this.initialized = true;

			// Reset warpedCount
			this.warpedCount = 0;
			
			// Create buffer arrays
			int size = width * height;
			this.bufWarpedResidual = new float[size];
			this.bufWarpedDx = new float[size];
			this.bufWarpedDy = new float[size];
			this.bufWarpedX = new float[size];
			this.bufWarpedY = new float[size];
			this.bufWarpedZ = new float[size];
			this.bufInvDepth = new float[size];
			this.bufInvDepthVariance = new float[size];
			this.bufWeightP = new float[size];
			
		}
	}
	
	
	/**
	 * Estimates the pose between a reference frame and a frame.
	 */
	@SuppressWarnings("static-access")
	SE3 trackFrame(TrackingReference referenceFrame, Frame frame, SE3 frameToRefInitialEstimate) {

		// Initialize
		initialize(frame.width(0), frame.height(0));
		
		// Initial estimate
		SE3 frameToRefEstimate = frameToRefInitialEstimate;
		SE3 refToFrame = SE3.inverse(frameToRefEstimate);
		
		System.out.println(Arrays.toString(SE3.ln(frameToRefEstimate)));
		System.out.println(Arrays.toString(SE3.ln(refToFrame)));
		
		// LS
		LGS6 ls = new LGS6();
		
		float lastResidual = 0;
		
		// For each pyramid level, coarse to fine
		for (int level=Constants.SE3TRACKING_MAX_LEVEL-1 ;
				level>=Constants.SE3TRACKING_MIN_LEVEL ;
				level-=1) {
			

			// Generate 3D points of reference frame for the level, if not already done.
			if (referenceFrame.pointCloudLvl[level] == null) {
				referenceFrame.pointCloudLvl[level] = referenceFrame.createPointCloud(
						referenceFrame.keyframe.inverseDepthLvl[level],
						referenceFrame.width(level), referenceFrame.height(level), level);
			}
			
			
			calculateResidualAndBuffers(referenceFrame, frame, refToFrame, level);
			
			
			// Diverge when amount of pixels successfully warped into new frame < some amount
			if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN * 
					frame.width(level)*frame.height(level)) {
				// Diverge
				System.out.println("Diverged.(1)");
				return null;
			}
			
			// Weighted SSD
			float lastError = calculateWeightsAndResidual(refToFrame);
			
			float LM_lambda = this.lambdaInitial[level];

			// For a maximum number of iterations
			for (int iteration=0 ; iteration < maxItsPerLvl[level] ; iteration++) {
				System.out.println("L" + level + " - " + iteration + " w/h " + frame.width(level)+"/"+frame.height(level));
				// Calculate/update LS
				calculateWarpUpdate(ls);
				
				int incTry = 0;
				while (true) {
					incTry++;
					
					// Solve LS to get increment
					jeigen.DenseMatrix inc = calcIncrement(ls, LM_lambda);

					// Apply increment
					SE3 newRefToFrame = SE3.exp(Vec.vec6ToArray(inc));
					newRefToFrame.mulEq(refToFrame);
					
					
					// Re-evaluate residual
					calculateResidualAndBuffers(referenceFrame, frame, newRefToFrame, level);
					
					
					// Check for divergence
					if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN * frame.width(level)*frame.height(level)) {
						// Diverge
						System.out.println("Diverged.(2)");
						System.out.println("warpedCount: " + warpedCount);
						return null;
					}
					
					// Calculate weighted residual/error
					float error = calculateWeightsAndResidual(newRefToFrame);
					
					if (error < lastError) {
						// Accept increment
						refToFrame = newRefToFrame;
						
//						System.out.println("inc:" + Arrays.toString(Vec.vecToArray(inc)) + LM_lambda);
//						System.out.println("vec6:" + Arrays.toString(SE3.ln(newRefToFrame)));
//						System.out.println("\n\n");
						
						// Check for convergence
						if (error / lastError > convergenceEps[level]) {
							// Stop iteration
							iteration = maxItsPerLvl[level];
						}
						lastError = error;
						lastResidual = error;
						
						// Update lambda
						if(LM_lambda <= 0.2) {
							LM_lambda = 0;
						} else {
							LM_lambda *= lambdaSuccessFac;
						}
						
						// Break!
						break;
					} else {
						double[] incVec = Vec.vec6ToArray(inc);
						double incVecDot = Vec.dot(incVec, incVec);
						if(!(incVecDot > stepSizeMin[level])) {
							// Stop iteration
							iteration = maxItsPerLvl[level];
							break;
						}
						
						// Update lambda
						if(LM_lambda == 0) {
							LM_lambda = 0.2f;
						} else {
							LM_lambda *= Math.pow(lambdaFailFac, incTry);
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
		

		jeigen.DenseMatrix rotationMat = refToFrame.getRotationMat();
		jeigen.DenseMatrix translationVec = refToFrame.getTranslationMat();
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
		
		
		System.out.println("Vec6: " + Arrays.toString(SE3.ln(refToFrame)));
		System.out.println("Rotation: " + refToFrame.getRotationMat());
		System.out.println("Translation: " + refToFrame.getTranslationMat());
		
		return refToFrame;
	}

	private jeigen.DenseMatrix calcIncrement(LGS6 ls, float LM_lambda) {
		jeigen.DenseMatrix b = new jeigen.DenseMatrix(ls.b.neg());
		jeigen.DenseMatrix A = new jeigen.DenseMatrix(ls.A);
		for (int i=0 ; i<6 ; i++) {
			//A(i,i) *= 1+LM_lambda;
			A.set(i, i, A.get(i, i) * (1 + LM_lambda));
		}
		jeigen.DenseMatrix inc = A.ldltSolve(b);
		return inc;
	}
	
	int calculateResidualAndBuffersCount = 0;
	/**
	 * Calculate residual and buffers
	 * 
	 * 
	 * 
	 * @param referenceFrame
	 * @param frame
	 * @param frameToRefPose
	 * @param level
	 * @return sum of un-weighted residuals, divided by good pixel count.
	 * 
	 */
	public float calculateResidualAndBuffers(TrackingReference referenceFrame,
			Frame frame, SE3 frameToRefPose, int level) {
		calculateResidualAndBuffersCount++;
		
		double fx = Constants.fx[level];
		double fy = Constants.fy[level];
		double cx = Constants.cx[level];
		double cy = Constants.cy[level];
		
		// Get rotation, translation matrix
		jeigen.DenseMatrix rotationMat = frameToRefPose.getRotationMat();
		jeigen.DenseMatrix translationVec = frameToRefPose.getTranslationMat();

		// TODO: For drawing image for debugging.
//		Mat debugImage = new Mat(referenceFrame.height(level), referenceFrame.width(level), CvType.CV_8UC1);
//		byte[] debugArray = new byte[(int) debugImage.total()];
//		debugImage.get(0, 0, debugArray);
		
		
		float sumResUnweighted = 0;
		
		int goodCount = 0;
		int badCount = 0;
		
		warpedCount = 0;
		
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
				// Skip this pixel
				continue;
			}
			
			
			// Interpolated intensity, gradient X,Y.
			float interpolatedIntensity = interpolatedValue(frame.imageArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientX = interpolatedValue(frame.imageGradientXArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientY = interpolatedValue(frame.imageGradientYArrayLvl[level], u, v, frame.width(level));
			
			
			// TODO: For drawing image for debugging.
//			debugArray[i] = (byte)interpolatedIntensity;
			
			float referenceFrameIntensity = (int) referenceFrame.keyframe.imageArrayLvl[level][i];
			
			//
			float c1 = referenceFrameIntensity;
			float c2 = interpolatedIntensity;
			float residual = c1 - c2;
			float squaredResidual = residual*residual;
			

			// Set buffers
			this.bufWarpedResidual[warpedCount] = residual;
			this.bufWarpedDx[warpedCount] = interpolatedGradientX;
			this.bufWarpedDy[warpedCount] = interpolatedGradientY;
			this.bufWarpedX[warpedCount] = (float) warpedPoint.get(0, 0);
			this.bufWarpedY[warpedCount] = (float) warpedPoint.get(1, 0);
			this.bufWarpedZ[warpedCount] = (float) warpedPoint.get(2, 0);
			this.bufInvDepth[warpedCount] = (float) (1.0f / point.get(2, 0));
			this.bufInvDepthVariance[warpedCount] = referenceFrame.keyframe.inverseDepthVarianceLvl[level][i];
			
			// Increase warpCount
			warpedCount += 1;
			
			// Condition related to gradient and residual, to determine if to
			// use the residual from this pixel or not.
			boolean isGood = squaredResidual / 
					(Constants.MAX_DIFF_CONSTANT + 
					 Constants.MAX_DIFF_GRAD_MULT * 
					 	(interpolatedGradientX*interpolatedGradientX + 
					 	 interpolatedGradientY*interpolatedGradientY)) < 1;
			
			if (isGood) {
				sumResUnweighted += squaredResidual;
				goodCount++;
			} else {
				badCount++;
			}
			
		}
		
		// TODO: For drawing image for debugging.
//		debugImage.put(0, 0, debugArray);
//		Highgui.imwrite("debugImage"+calculateResidualAndBuffersCount+"-"+ sumResUnweighted/goodCount + ".jpg", debugImage);
		
		return sumResUnweighted / goodCount;
	}
	
	/**
	 * calcWeightsAndResidual
	 * 
	 * @param referenceToFrame
	 * @return sum of weighted residuals divided by warpedCount
	 */
	public float calculateWeightsAndResidual(SE3 referenceToFrame) {
		
		float tx = (float) referenceToFrame.getTranslation()[0];
		float ty = (float) referenceToFrame.getTranslation()[1];
		float tz = (float) referenceToFrame.getTranslation()[2];

		float sumRes = 0;

		for(int i=0 ; i<warpedCount ; i++) {
			float px = bufWarpedX[i];	// x'
			float py = bufWarpedY[i];	// y'
			float pz = bufWarpedZ[i];	// z'
			float d  = bufInvDepth[i];	// d
			float rp = bufWarpedResidual[i]; // r_p
			float gx = bufWarpedDx[i];		// \delta_x I
			float gy = bufWarpedDy[i];  	// \delta_y I
			float s  = varWeight * bufInvDepthVariance[i];	// \sigma_d^2
			
			// calc dw/dd (first 2 components):
			float g0 = (tx * pz - tz * px) / (pz*pz*d);
			float g1 = (ty * pz - tz * py) / (pz*pz*d);

			// calc w_p
			float drpdd = gx * g0 + gy * g1;	// ommitting the minus
			float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);

			float weighted_rp = (float) Math.abs(rp * Math.sqrt(w_p));

			float wh = Math.abs(weighted_rp < (huberD/2f) ?
					1 : (huberD/2f) / weighted_rp);

			sumRes += wh * w_p * rp*rp;
			
			// Set weight into buffer
			this.bufWeightP[i] = wh * w_p;
		}

		return sumRes / warpedCount;
	}
	
	/**
	 * calculateWarpUpdate
	 * @param ls
	 */
	public void calculateWarpUpdate(LGS6 ls) {
		
		ls.initialize();
		
		// For each warped pixel
		for (int i=0 ; i<warpedCount ; i++) {

			// x,y,z
			float px = bufWarpedX[i];
			float py = bufWarpedY[i];
			float pz = bufWarpedZ[i];
			// Residual
			float r =  bufWarpedResidual[i];
			// Gradient
			float gx = bufWarpedDx[i];
			float gy = bufWarpedDy[i];

			// inverse depth
			float z = 1.0f / pz;
			float z_sqr = 1.0f / (pz*pz);
			
			// Vector6
			jeigen.DenseMatrix v = new jeigen.DenseMatrix(new double[][]{
					{z*gx},
					{z*gy},
					{(-px * z_sqr) * gx + (-py * z_sqr) * gy},
					{(-px * py * z_sqr) * gx + (-(1.0 + py * py * z_sqr)) * gy},
					{(1.0 + px * px * z_sqr) * gx + (px * py * z_sqr) * gy},
					{(-py * z) * gx + (px * z) * gy}});		
			
			// Integrate into A and b
			ls.update(v, r, bufWeightP[i]);
			
		}
	
		// Solve LS
		ls.finish();
		
	}
	
	
	// Entry point used to test Tracker
	public static void main(String[] args) {
		
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		Constants.setK(500, 500, 640/2, 480/2);
		
		
		// Read image
		Mat image1 = null;
		Mat image2 = null;
		image1 = Highgui.imread("test1.jpg");
		image2 = Highgui.imread("test2.jpg");
		
		// Convert to grayscale
		Imgproc.cvtColor(image1, image1, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(image2, image2, Imgproc.COLOR_RGB2GRAY);
		
		System.out.println("Image sizes: " +  image1.width() + ", " + image1.height());
		
		
		Frame frame1 = new Frame(image1);
		Frame frame2 = new Frame(image2);
		TrackingReference refFrame = new TrackingReference(frame1);
		
		
		// Track frame
		Tracker tracker = new Tracker();
		SE3 se3 = tracker.trackFrame(refFrame, frame2, SE3.exp(new double[]{0,0,0,0,0,0}));
		
		
		
	}
	
	
	static float interpolatedValue(byte[] dataArray, double x, double y, int width) {

		int ix = (int)x;
		int iy = (int)y;
		float dx = (float) (x - ix);
		float dy = (float) (y - iy);
		float dxdy = dx*dy;
		int bp = ix+iy*width;
		
		float res =   dxdy 			* (float)((int)dataArray[bp+1+width])
					+ (dy-dxdy) 	* (float)((int)dataArray[bp+width])
					+ (dx-dxdy) 	* (float)((int)dataArray[bp+1])
					+ (1-dx-dy+dxdy)* (float)((int)dataArray[bp]);

		return res;
		
	}
	
	static float interpolatedValue(float[] dataArray, double x, double y, int width) {

		int ix = (int)x;
		int iy = (int)y;
		float dx = (float) (x - ix);
		float dy = (float) (y - iy);
		float dxdy = dx*dy;
		int bp = ix+iy*width;
		
		float res =   dxdy 			* (float)((int)dataArray[bp+1+width])
					+ (dy-dxdy) 	* (float)((int)dataArray[bp+width])
					+ (dx-dxdy) 	* (float)((int)dataArray[bp+1])
					+ (1-dx-dy+dxdy)* (float)((int)dataArray[bp]);

		return res;
		
	}
	
}
