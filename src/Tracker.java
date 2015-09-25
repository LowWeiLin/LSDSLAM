import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.Arrays;

import jeigen.DenseMatrix;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import DataStructures.Frame;
import DataStructures.TrackingReference;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import LieAlgebra.Vec;
import Utils.Constants;
import Utils.Utils;



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
	
	float pointUsage = 0;
	float lastGoodCount = 0;
	float lastMeanRes = 0;
	float lastBadCount = 0;
	float lastResidual = 0;
	
	boolean trackingWasGood = false;
	boolean diverged = false;
	
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
		
		
		// **
		// Do not use more than 4 levels for odometry tracking
		for (int level = 4; level < Constants.PYRAMID_LEVELS; ++level)
			maxItsPerLvl[level] = 0;

		
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

		diverged = false;
		trackingWasGood = true;
		
		// Initialize
		initialize(frame.width(0), frame.height(0));
		
		// Initial estimate
		SE3 frameToRefEstimate = frameToRefInitialEstimate;
		SE3 refToFrame = SE3.inverse(frameToRefEstimate);
		
		//System.out.println("Init: "+Arrays.toString(SE3.ln(frameToRefEstimate)));
		System.out.println("---tracking "+ frame.id() +" ---");
		System.out.println("Init refToFrame: "+Arrays.toString(SE3.ln(refToFrame)));
		
		// LS
		LGS6 ls = new LGS6();
		
		float lastResidual = 0;
		
		// For each pyramid level, coarse to fine
		for (int level=Constants.SE3TRACKING_MAX_LEVEL-1 ;
				level>=Constants.SE3TRACKING_MIN_LEVEL ;
				level-=1) {
			

			// Generate 3D points of reference frame for the level, if not already done.
			//if (referenceFrame.pointCloudLvl[level] == null) {
				referenceFrame.pointCloudLvl[level] = 
						referenceFrame.createPointCloud(
						referenceFrame.keyframe.inverseDepthLvl[level],
						referenceFrame.keyframe.inverseDepthVarianceLvl[level],
						referenceFrame.width(level), referenceFrame.height(level),
						level);
			//}
			
			// TODO: Write point cloud to file
			if (level == 1) {
				try {
					TrackingReference.writePointCloudToFile("pointCloud-"+frame.id()+"-"+level+".xyz",
							referenceFrame.pointCloudLvl[level], referenceFrame.width(level), referenceFrame.height(level));
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
//			try {
//				TrackingReference.writePointCloudToFile("pointCloud-"+frame.id()+"-"+level+".xyz",
//						referenceFrame);
//			} catch (FileNotFoundException | UnsupportedEncodingException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
			
			
			calculateResidualAndBuffers(referenceFrame,
										frame,
										refToFrame,
										level);
			
			// Diverge when amount of pixels successfully warped into new frame < some amount
			if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN * 
					frame.width(level)*frame.height(level)) {
				// Diverge
				System.out.println("Diverged.(1)");
				diverged = true;
				trackingWasGood = false;
				return null;
			}
			
			// Weighted SSD
			float lastError = calculateWeightsAndResidual(refToFrame);
			
			float LM_lambda = this.lambdaInitial[level];

			// TODO: check again
			System.out.println("L" + level + " - " + "last err" + " " + lastError);
			
			// For a maximum number of iterations
			for (int iteration=0 ; iteration < maxItsPerLvl[level] ; iteration++) {
				// Calculate/update LS
				calculateWarpUpdate(ls);
				
				int incTry = 0;
				while (true) {
					incTry++;
					
					// Solve LS to get increment
					jeigen.DenseMatrix inc = calcIncrement(ls, LM_lambda);
					//System.out.println(incTry + " : " + LM_lambda + inc);
					
					// Apply increment
					SE3 newRefToFrame = SE3.exp(Vec.vec6ToArray(inc));
					newRefToFrame.mulEq(refToFrame);

					//System.out.println("Applied inc " + Arrays.toString(SE3.ln(newRefToFrame)));
					
					
					// Re-evaluate residual
					calculateResidualAndBuffers(referenceFrame,
							frame,
							newRefToFrame,
							level);
					
					
					// Check for divergence
					if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN * frame.width(level)*frame.height(level)) {
						// Diverge
						System.out.println("Diverged.(2)");
						System.out.println("warpedCount: " + warpedCount);
						diverged = true;
						trackingWasGood = false;
						return null;
					}
					
					//System.out.println("newRefToFrame " + Arrays.toString(SE3.ln(newRefToFrame)));
					
					// Calculate weighted residual/error
					float error = calculateWeightsAndResidual(newRefToFrame);
					//System.out.println("error: " + error);
					if (error < lastError) {
						//System.out.println("error < lastError");
						// Accept increment
						refToFrame = newRefToFrame;
						
//						System.out.println("inc:" + Arrays.toString(Vec.vecToArray(inc)) + LM_lambda);
//						System.out.println("vec6:" + Arrays.toString(SE3.ln(newRefToFrame)));
//						System.out.println("\n\n");
						
						// Check for convergence
						if (error / lastError > convergenceEps[level]) {
							// Stop iteration
							iteration = maxItsPerLvl[level];
							//System.out.println("Converged");
						}
						lastError = error;
						lastResidual = error;
						
						// Update lambda
						if(LM_lambda <= 0.2) {
							LM_lambda = 0;
						} else {
							LM_lambda *= lambdaSuccessFac;
						}
						//System.out.println("break");
						// Break!
						break;
					} else {
						double[] incVec = Vec.vec6ToArray(inc);
						double incVecDot = Vec.dot(incVec, incVec);
						if(!(incVecDot > stepSizeMin[level])) {
							// Stop iteration
							iteration = maxItsPerLvl[level];
							//System.out.println("Step size below min");
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

				//System.out.println("L" + level + " - " + iteration + " " + lastError);
			}
			
			
		}
		

		trackingWasGood = !diverged
				&& lastGoodCount / (frame.width(Constants.SE3TRACKING_MIN_LEVEL)*frame.height(Constants.SE3TRACKING_MIN_LEVEL)) > Constants.MIN_GOODPERALL_PIXEL
				&& lastGoodCount / (lastGoodCount + lastBadCount) > Constants.MIN_GOODPERGOODBAD_PIXEL;

		if(trackingWasGood)
			referenceFrame.keyframe.numFramesTrackedOnThis++;

		
		// TODO: Use dummy pose to test mapping
		//refToFrame = SE3.inverse(SE3.exp(dummy[f%15]));
		//f++;
		
		
		
		
		frame.initialTrackedResidual = lastResidual / pointUsage;
		frame.pose.thisToParent_raw = new SIM3(SE3.inverse(refToFrame), 1);
		frame.pose.trackingParent = referenceFrame.keyframe.pose;
		

		// Test writing original 3D point cloud
//		try {
//			referenceFrame.writePointCloudToFile("out1-"+frame.id()+".xyz", referenceFrame.pointCloudLvl[0],
//					referenceFrame.width(0), referenceFrame.height(0));
//		} catch (FileNotFoundException | UnsupportedEncodingException e) {
//			e.printStackTrace();
//		}
		

		// Offset 3D point cloud by refToFrame
//		jeigen.DenseMatrix rotationMat = refToFrame.getRotationMat();
//		jeigen.DenseMatrix translationVec = refToFrame.getTranslationMat();
//		for (int i=0 ; i<referenceFrame.pointCloudLvl[0].length ; i++) {
//			// Each 3D point
//			jeigen.DenseMatrix point = referenceFrame.pointCloudLvl[0][i];
//			
//			// Skip if point is not valid
//			if (point == null) {
//				continue;
//			}
//			
//			// Warp to 2D image by estimate
//			jeigen.DenseMatrix warpedPoint = rotationMat.mmul(point).add(translationVec);
//			referenceFrame.pointCloudLvl[0][i] = warpedPoint;
//		}
		
		// Test writing offset 3D point cloud
//		try {
//			referenceFrame.writePointCloudToFile("out2.xyz", referenceFrame.pointCloudLvl[0],
//					referenceFrame.width(0), referenceFrame.height(0));
//		} catch (FileNotFoundException | UnsupportedEncodingException e) {
//			e.printStackTrace();
//		}
		
		
		System.out.println("Final frameToRef: " + Arrays.toString(SE3.ln(SE3.inverse(refToFrame))));

		//System.out.println("Rotation: " + refToFrame.getRotationMat());
		//System.out.println("Translation: " + refToFrame.getTranslationMat());
		
		return SE3.inverse(refToFrame);
	}
	public int f = 0;
	public static final double[][] dummy = new double[][]{
		{-0.00175238851,-0.00469227021,0.000993556356,-0.00457633686,.00256084883,0.00164334879},
		{-0.0032370568,-0.0071266297,0.000131566101,-0.00587451305,0.00416550569,0.00175637383},
		{-0.00524847961,-0.00916995613,0.00073871208,-0.00682622829,0.00620494594,0.00117302331},
		{-0.010111267,-0.0122120611,-0.00222993763,-0.00959468307,0.0112431362,0.00128104885},
		{-0.0133069071,-0.0130967127,-0.00253995195,-0.0109903534,0.0132610342,0.000120093354},
		{-0.0164450003,-0.0136377944,-0.00158586493,-0.0120422609,0.0156781726,-0.00115543701},
		{-0.0160342665,-0.0123697938,0.000578902925,-0.0110716074,0.0174666422,-0.000227941858},
		{-0.0221007355,-0.0191942833,-0.00251983725,-0.017727924,0.0257640394,0.00339003636},
		{-0.0281898677,-0.0237951226,-0.00777249248,-0.0240689471,0.0305052818,0.00320009437},
		{-0.040371606,-0.0227181611,-0.00656240021,-0.0272613005,0.037595325,-0.000599596067},
		{-0.0428053593,-0.0228121154,-0.00198295719,-0.0279127882,0.0385276832,-0.00136081388},
		{-0.0450458789,-0.0266367741,-0.00326801571,-0.0309348887,0.0405216672,-0.00154802265},
		{-0.0470093594,-0.0258653192,-0.000641825914,-0.0299376692,0.0420608524,-0.00362433593},
		{-0.0477334866,-0.0306360378,0.000595234146,-0.0330749078,0.0437194545,-0.00495575213},
		{-0.0509633143,-0.0352358335,0.0015304195,-0.0365532545,0.0479359235,-0.00584868922}
	};
	

	private jeigen.DenseMatrix calcIncrement(LGS6 ls, float LM_lambda) {
		jeigen.DenseMatrix b = new jeigen.DenseMatrix(ls.b.neg());
		jeigen.DenseMatrix A = new jeigen.DenseMatrix(ls.A);
		for (int i=0 ; i<6 ; i++) {
			//A(i,i) *= 1+LM_lambda;
			A.set(i, i, A.get(i, i) * (1 + LM_lambda));
		}
		jeigen.DenseMatrix inc = A.ldltSolve(b);
		//System.out.println("A:\n" + A + "\nb\n" + b);
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
		
		//System.out.println("F: " + frame.id() + " L: " + level);
		
		calculateResidualAndBuffersCount++;
		
		double fx = Constants.fx[level];
		double fy = Constants.fy[level];
		double cx = Constants.cx[level];
		double cy = Constants.cy[level];
		
		//System.out.println(fx + " " + fy + " " + cx + " " + cy);
		
		// Get rotation, translation matrix
		jeigen.DenseMatrix rotationMat = frameToRefPose.getRotationMat();
		jeigen.DenseMatrix translationVec = frameToRefPose.getTranslationMat();

		//System.out.println("R: " + rotationMat);
		//System.out.println("T: " + translationVec);

		//System.out.println("wh: " + frame.width(level) + " " + frame.height(level));
		
		// TODO: For drawing image for debugging.
//		Mat debugImage = new Mat(referenceFrame.height(level), referenceFrame.width(level), CvType.CV_8UC1);
//		byte[] debugArray = new byte[(int) debugImage.total()];
//		debugImage.get(0, 0, debugArray);
		
		
		float sumResUnweighted = 0;
		
		boolean[] isGoodOutBuffer = Constants.SE3TRACKING_MIN_LEVEL == level ? frame.refPixelWasGood() : null;
		
		int goodCount = 0;
		int badCount = 0;

		// Sum of residuals. Used for what?
		float sumSignedRes = 0;
		
		// what?
		float usageCount = 0;
		
		warpedCount = 0;
		
		
		int numValidPoints = 0;
		
		int inImage = 0;
		
		// For each point in point cloud
		for (int i=0 ; i<referenceFrame.pointCloudLvl[level].length ; i++) {
			// 3D position
			jeigen.DenseMatrix point = referenceFrame.pointCloudLvl[level][i];
			
			// Skip if point is not valid
			if (point == null) {
				continue;
			} else {
				numValidPoints++;
			}
			
			
			
			// Warp to 2D image by estimate
			jeigen.DenseMatrix warpedPoint = rotationMat.mmul(point).add(translationVec);
			
//			System.out.println("point: " + point);
//			System.out.println("wxp: " + warpedPoint);
//			DecimalFormat df = new DecimalFormat("#0.0000000000000");
//			System.out.println("wxp: " + df.format(warpedPoint.get(0, 0)) + " "
//									   + df.format(warpedPoint.get(1, 0)) + " "
//									   + df.format(warpedPoint.get(2, 0)));
			
			// Image points
			double u = (warpedPoint.get(0, 0)/warpedPoint.get(2, 0))*fx + cx;
			double v = (warpedPoint.get(1, 0)/warpedPoint.get(2, 0))*fy + cy;
			
			//System.out.println("uv: " + u + " " + v);
			
			// Check image points within bounds
			if (!(u>1 && v>1 && u<frame.width(level)-2 && v<frame.height(level)-2)) {
				if(isGoodOutBuffer != null)
					isGoodOutBuffer[i] = false;
				// Skip this pixel
				//System.out.println("-uv: " + df.format(u) + " " + df.format(v));
				continue;
			}
			//System.out.println("+uv: " + df.format(u) + " " + df.format(v));
			
			inImage++;
			
//			System.out.println(i);
//			System.out.println("--\nP: \n" + point);
//			System.out.println("R: \n" + rotationMat);
//			System.out.println("T: \n" + translationVec);

			// Interpolated intensity, gradient X,Y.
			float interpolatedIntensity = Utils.interpolatedValue(frame.imageArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientX = Utils.interpolatedValue(frame.imageGradientXArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientY = Utils.interpolatedValue(frame.imageGradientYArrayLvl[level], u, v, frame.width(level));
			
			
			// TODO: For drawing image for debugging.
//			debugArray[i] = (byte)interpolatedIntensity;
			
			float referenceFrameIntensity = referenceFrame.keyframe.imageArrayLvl[level][i];
			
			//
			float c1 = referenceFrameIntensity;
			float c2 = interpolatedIntensity;
			float residual = c1 - c2;
			float squaredResidual = residual*residual;
			
			//System.out.println("l: " + level);
			//System.out.println("r: " + c1 + " " + c2);
			//System.out.println("Redidual: " + residual);

			// Set buffers
			this.bufWarpedResidual[warpedCount] = residual;
			
			//** Gradient multiplied by fx fy?
			this.bufWarpedDx[warpedCount] = (float) (fx * interpolatedGradientX);
			this.bufWarpedDy[warpedCount] = (float) (fy * interpolatedGradientY);
			
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
			

			if(isGoodOutBuffer != null)
				isGoodOutBuffer[i] = isGood;
			
			if (isGood) {
				sumResUnweighted += squaredResidual;
				sumSignedRes += residual;
				goodCount++;
			} else {
				badCount++;
			}
			

			// Change in depth
			float depthChange = (float) (point.get(2,0) / warpedPoint.get(2, 0));	// if depth becomes larger: pixel becomes "smaller", hence count it less.
			// What? pixels used?
			usageCount += depthChange < 1 ? depthChange : 1;

			
		}
		
		// TODO: For drawing image for debugging.
//		debugImage.put(0, 0, debugArray);
//		Highgui.imwrite("debugImage"+calculateResidualAndBuffersCount+"-"+ sumResUnweighted/goodCount + ".jpg", debugImage);
		

		pointUsage = usageCount / (float)numValidPoints;
		lastGoodCount = goodCount;
		lastBadCount = badCount;
		lastMeanRes = sumSignedRes / goodCount;
		
		/*
		System.out.println("calcResidualAndBuffers lvl" + level);
		

		System.out.println("inImage: " + inImage);
		System.out.println("numValid: " + numValidPoints);
		System.out.println("calcResidualAndBuffers sumUnweighted " + sumResUnweighted);
		System.out.println("calcResidualAndBuffers good " + goodCount);

		System.out.println("calcResidualAndBuffers " + sumResUnweighted / goodCount);
		*/
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
		
		//System.out.println("warpedCount: " + warpedCount );
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
			
			//System.out.println("xyz: " + px + "," + py + "," + pz);
			//System.out.println("r: " + r);
			//System.out.println("gxy: " + gx + "," + gy);
			
			//System.out.println("V:"+v+"\nR:"+r);
			// Integrate into A and b
			ls.update(v, r, bufWeightP[i]);
			
		}
	
		// Solve LS
		ls.finish();
		
	}
	
	
	// Entry point 1 used to test Tracker
	public static void main2(String[] args) {
		
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
	
	// Entry point 2 used to test Tracker
	public static void main(String[] args) {
		
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		Constants.setK(500, 500, 640/2, 480/2);
		
		
		VideoCapture capture = new VideoCapture();
		capture.open("table-vid-trimmed.avi");
		Mat mat = new Mat();
		
		Mat firstFrame = new Mat();
		capture.read(firstFrame);
		Imgproc.cvtColor(firstFrame, firstFrame, Imgproc.COLOR_RGB2GRAY);

		Frame frame1 = new Frame(firstFrame);
		TrackingReference refFrame = new TrackingReference(frame1);
		
		Tracker tracker = new Tracker();
		
		
		// Write 3D points of camera center
		PrintWriter writer = null;
		try {
			writer = new PrintWriter("3dpoints.xyz", "ASCII");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		writer.println(3);
		
		int count = 0;
		for(;;){
			//System.out.println(count);
			boolean success = capture.read(mat); //reads captured frame into the Mat image
			
			if (!success || count >= 50) {
				break;
			}
			Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);
			   
			Frame frame = new Frame(mat);
			
			//Highgui.imwrite("Test-"+count+".jpg", frame.imageLvl[0]);
			
			System.out.println(refFrame.keyframe.id()+"-->"+frame.id());
			
			
			SE3 se3 = tracker.trackFrame(refFrame, frame, SE3.exp(new double[]{0,0,0,0,0,0}));
			
			if (se3 == null) // Diverged
				continue;
			
			System.out.println("Vec6: " + Arrays.toString(SE3.ln(se3)));
			   
			double[] se3Array = SE3.ln(se3);
			
			writer.printf("%.6f ", se3Array[0]);
			writer.printf("%.6f ", se3Array[1]);
			writer.printf("%.6f\n", se3Array[2]);
			
			writer.flush();
			
			
			count++;
			
		}
		
		
		writer.close();
		
		
	}
	
	
	
	
}
