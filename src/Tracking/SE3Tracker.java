package Tracking;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

import jeigen.DenseMatrix;
import DataStructures.Frame;
import DataStructures.TrackingReference;
import IO.KeyFrameDisplay;
import LieAlgebra.SE3;
import LieAlgebra.SIM3;
import LieAlgebra.Vec;
import Utils.Constants;
import Utils.Utils;

public class SE3Tracker {
	
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
	
	

	public static final float lambdaInitialTestTrack = 0f;
	public static final float stepSizeMinTestTrack = 1e-3f;
	public static final float convergenceEpsTestTrack = 0.98f;
	public static final int maxItsTestTrack = 5;
	
	
	// Variables set when tracking
	int warpedCount = 0; // Number of pixels warped into new image bounds
	
	public float pointUsage = 0;
	public float lastGoodCount = 0;
	public float lastMeanRes = 0;
	public float lastBadCount = 0;
	public float lastResidual = 0;
	
	public boolean trackingWasGood = false;
	public boolean diverged = false;
	
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
	
	
	
	public SE3Tracker() {
		
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
	public SE3 trackFrame(TrackingReference referenceFrame, Frame frame, SE3 frameToRefInitialEstimate) {

		diverged = false;
		trackingWasGood = true;
		
		// Initialize
		initialize(frame.width(0), frame.height(0));
		
		// Initial estimate
		SE3 frameToRefEstimate = frameToRefInitialEstimate;
		SE3 refToFrame = SE3.inverse(frameToRefEstimate);
		
		
		System.out.println("---tracking "+ frame.id() +" ---");
		System.out.println("Init frameToRefInitialEstimate: "+Arrays.toString(SE3.ln(frameToRefInitialEstimate)));
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
					referenceFrame.createPointCloud(level);
			//}
			
			// TODO: Write point cloud to file
			if (level == 1) {
				try {
					KeyFrameDisplay kfd = new KeyFrameDisplay(referenceFrame.keyframe);
					kfd.writePointCloudToFile("KFpointCloud-"+frame.id()+"-"+level+".ply");
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					e.printStackTrace();
				}
			}
			
			calculateResidualAndBuffers(referenceFrame.posDataLvl[level],
					referenceFrame.colorAndVarDataLvl[level],
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
					calculateResidualAndBuffers(referenceFrame.posDataLvl[level],
							referenceFrame.colorAndVarDataLvl[level],
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
					
					// Calculate weighted residual/error
					float error = calculateWeightsAndResidual(newRefToFrame);
					if (error < lastError) {
						// Accept increment
						refToFrame = newRefToFrame;
						
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
			}
			
			
		}
		

		trackingWasGood = !diverged
				&& lastGoodCount / (frame.width(Constants.SE3TRACKING_MIN_LEVEL)*frame.height(Constants.SE3TRACKING_MIN_LEVEL))
				> Constants.MIN_GOODPERALL_PIXEL
				&& lastGoodCount / (lastGoodCount + lastBadCount) > Constants.MIN_GOODPERGOODBAD_PIXEL;

		if(trackingWasGood)
			referenceFrame.keyframe.numFramesTrackedOnThis++;
		else {
			System.out.println("Tracking was bad ??");
			System.out.println("lastGoodCount " + lastGoodCount);
			System.out.println("lastBadCount " + lastBadCount);
			System.out.println("size: " + (frame.width(Constants.SE3TRACKING_MIN_LEVEL)*frame.height(Constants.SE3TRACKING_MIN_LEVEL)));
		}
		

		SE3 frameToRef = SE3.inverse(refToFrame);

		frame.initialTrackedResidual = lastResidual / pointUsage;
		frame.pose.thisToParent_raw = new SIM3(frameToRef, 1.0);
		frame.pose.trackingParent = referenceFrame.keyframe.pose;
		
		// Add frameToRef to reference frame
		referenceFrame.keyframe.trackedOnPoses.add(frameToRef);

		System.out.println("Final frameToRef: " + Arrays.toString(SE3.ln(frameToRef)));
		
		return frameToRef;
	}
	
	private jeigen.DenseMatrix calcIncrement(LGS6 ls, float LM_lambda) {
		jeigen.DenseMatrix b = new jeigen.DenseMatrix(ls.b.neg());
		jeigen.DenseMatrix A = new jeigen.DenseMatrix(ls.A);
		for (int i=0 ; i<6 ; i++) {
			A.set(i, i, A.get(i, i) * (1 + LM_lambda)); //A(i,i) *= 1+LM_lambda;
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
	public float calculateResidualAndBuffers(
			DenseMatrix[] posData,
			DenseMatrix[] colorAndVarData,
			Frame frame, SE3 frameToRefPose, int level) {
		
		calculateResidualAndBuffersCount++;
		
		double fx = Constants.fx[level];
		double fy = Constants.fy[level];
		double cx = Constants.cx[level];
		double cy = Constants.cy[level];
		
		
		// Get rotation, translation matrix
		jeigen.DenseMatrix rotationMat = frameToRefPose.getRotationMat();
		jeigen.DenseMatrix translationVec = frameToRefPose.getTranslationMat();

		float sumResUnweighted = 0;
		
		boolean[] isGoodOutBuffer = Constants.SE3TRACKING_MIN_LEVEL == level ? frame.refPixelWasGood() : null;
		
		int goodCount = 0;
		int badCount = 0;

		float sumSignedRes = 0;
		
		// what?
		float usageCount = 0;
		
		warpedCount = 0;
		
		
		int numValidPoints = 0;
		//int inImage = 0;
		
		// For each point in point cloud
		for (int i=0 ; i<posData.length ; i++) {
			// 3D position
			jeigen.DenseMatrix point = posData[i];
			
			// Skip if point is not valid
			if (point == null) {
				continue;
			} else {
				numValidPoints++;
			}
			
			// Warp to 2D image by estimate
			jeigen.DenseMatrix warpedPoint = rotationMat.mmul(point).add(translationVec);
		
			// Image points
			double u = (warpedPoint.get(0, 0)/warpedPoint.get(2, 0))*fx + cx;
			double v = (warpedPoint.get(1, 0)/warpedPoint.get(2, 0))*fy + cy;
			
			// Check image points within bounds
			if (!(u>1 && v>1 && u<frame.width(level)-2 && v<frame.height(level)-2)) {
				if(isGoodOutBuffer != null)
					isGoodOutBuffer[i] = false;
				// Skip this pixel
				continue;
			}
			//inImage++;
			
			// Interpolated intensity, gradient X,Y.
			float interpolatedIntensity = Utils.interpolatedValue(frame.imageArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientX = Utils.interpolatedValue(frame.imageGradientXArrayLvl[level], u, v, frame.width(level));
			float interpolatedGradientY = Utils.interpolatedValue(frame.imageGradientYArrayLvl[level], u, v, frame.width(level));
			
			float referenceFrameIntensity = (float) colorAndVarData[i].get(0, 0);
			
			//
			float c1 = referenceFrameIntensity;
			float c2 = interpolatedIntensity;
			float residual = c1 - c2;
			float squaredResidual = residual*residual;
	
			// Set buffers
			this.bufWarpedResidual[warpedCount] = residual;
			
			this.bufWarpedDx[warpedCount] = (float) (fx * interpolatedGradientX);
			this.bufWarpedDy[warpedCount] = (float) (fy * interpolatedGradientY);
			
			this.bufWarpedX[warpedCount] = (float) warpedPoint.get(0, 0);
			this.bufWarpedY[warpedCount] = (float) warpedPoint.get(1, 0);
			this.bufWarpedZ[warpedCount] = (float) warpedPoint.get(2, 0);
			this.bufInvDepth[warpedCount] = (float) (1.0f / point.get(2, 0));
			this.bufInvDepthVariance[warpedCount] = (float) colorAndVarData[i].get(1, 0);
			
			
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
			// Pixels used?
			usageCount += depthChange < 1 ? depthChange : 1;

		}
		
		pointUsage = usageCount / (float)numValidPoints;
		lastGoodCount = goodCount;
		lastBadCount = badCount;
		lastMeanRes = sumSignedRes / goodCount;
		
//		System.out.println("calcResidualAndBuffers lvl"+ level);
//		System.out.println("inImage " + inImage);
//		System.out.println("numValid " + numValidPoints);
//		System.out.println("calcResidualAndBuffers sumUnweighted " + sumResUnweighted);
//		System.out.println("calcResidualAndBuffers good " + goodCount);
//
//		System.out.println("calcResidualAndBuffers " + sumResUnweighted / goodCount);
		
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
	
	public float checkPermaRefOverlap(Frame reference, SE3 referenceToFrameOrg)
	{
		SE3 referenceToFrame = referenceToFrameOrg;
		
		int w2 = reference.width(Constants.QUICK_KF_CHECK_LVL)-1;
		int h2 = reference.height(Constants.QUICK_KF_CHECK_LVL)-1;
		DenseMatrix KLvl = Constants.K[Constants.QUICK_KF_CHECK_LVL];
		float fx_l = (float) KLvl.get(0,0);
		float fy_l = (float) KLvl.get(1,1);
		float cx_l = (float) KLvl.get(0,2);
		float cy_l = (float) KLvl.get(1,2);

		DenseMatrix rotMat = referenceToFrame.getRotationMat();
		DenseMatrix transVec = referenceToFrame.getTranslationMat();

		DenseMatrix[] refPoint = reference.permaRef_posData;
		
		float usageCount = 0;
		for(int i=0 ; i<refPoint.length ; i++) {
			
			if (refPoint[i] == null) {
				continue;
			}
			DenseMatrix point = refPoint[i];
			
			DenseMatrix Wxp = rotMat.mmul(point).add(transVec);
			float u_new = (float) ((Wxp.get(0, 0)/Wxp.get(2, 0))*fx_l + cx_l);
			float v_new = (float) ((Wxp.get(1, 0)/Wxp.get(2, 0))*fy_l + cy_l);
			if((u_new > 0 && v_new > 0 && u_new < w2 && v_new < h2))
			{
				float depthChange = (float) ((point).get(2, 0) / Wxp.get(2, 0));
				usageCount += depthChange < 1 ? depthChange : 1;
			}
		}

		pointUsage = usageCount / (float)reference.permaRefNumPts;
		return pointUsage;
	}

	public SE3 trackFrameOnPermaref(Frame reference, Frame frame, SE3 referenceToFrameOrg) {

			SE3 referenceToFrame = referenceToFrameOrg;

			LGS6 ls = new LGS6();
			diverged = false;
			trackingWasGood = true;

			calculateResidualAndBuffers(reference.permaRef_posData,
					reference.permaRef_colorAndVarData,
					frame,
					referenceToFrame,
					Constants.QUICK_KF_CHECK_LVL);
			
			// Number of pixels warped into new image < some amount (1%) ?
			if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN * 
					(frame.width(0)>>Constants.QUICK_KF_CHECK_LVL)*
					(frame.height(0)>>Constants.QUICK_KF_CHECK_LVL))
			{
				// diverged/lost tracking?
				diverged = true;
				trackingWasGood = false;
				System.out.println("Tracking bad warped count " + warpedCount);
				return new SE3();
			}
			
			float lastErr = calculateWeightsAndResidual(referenceToFrame);

			float LM_lambda = lambdaInitialTestTrack;

			for(int iteration=0; iteration < maxItsTestTrack; iteration++)
			{
				calculateWarpUpdate(ls);


				int incTry=0;
				while(true)
				{
					// solve LS system with current lambda
					jeigen.DenseMatrix inc = calcIncrement(ls, LM_lambda);
					incTry++;

					// apply increment. pretty sure this way round is correct, but hard to test.
					SE3 new_referenceToFrame = SE3.exp(Vec.vec6ToArray(inc)).mul(referenceToFrame);

					// re-evaluate residual
					calculateResidualAndBuffers(reference.permaRef_posData,
							reference.permaRef_colorAndVarData,
							frame,
							new_referenceToFrame,
							Constants.QUICK_KF_CHECK_LVL);
							
					if(warpedCount < Constants.MIN_GOODPERALL_PIXEL_ABSMIN *
							(frame.width(0)>>Constants.QUICK_KF_CHECK_LVL)*(frame.height(0)>>Constants.QUICK_KF_CHECK_LVL))
					{
						diverged = true;
						trackingWasGood = false;
						return new SE3();
					}
					float error = calculateWeightsAndResidual(new_referenceToFrame);


					// accept inc?
					if(error < lastErr)
					{
						// accept inc
						referenceToFrame = new_referenceToFrame;
						
						// converged?
						if(error / lastErr > convergenceEpsTestTrack)
							iteration = maxItsTestTrack;


						lastErr = error;


						if(LM_lambda <= 0.2)
							LM_lambda = 0;
						else
							LM_lambda *= lambdaSuccessFac;

						// Breaks while loop

						break;
					}
					else
					{

						double[] incVec = Vec.vec6ToArray(inc);
						double incVecDot = Vec.dot(incVec, incVec);
						if(!(incVecDot > stepSizeMinTestTrack)) {
							iteration = maxItsTestTrack;

							// Breaks while loop
							break;
						}

						if(LM_lambda == 0)
							LM_lambda = 0.2f;
						else
							LM_lambda *= Math.pow(lambdaFailFac, incTry);
					}
				}
			}

			lastResidual = lastErr;

			trackingWasGood = !diverged
					&& lastGoodCount / (frame.width(Constants.QUICK_KF_CHECK_LVL)*frame.height(Constants.QUICK_KF_CHECK_LVL)) > Constants.MIN_GOODPERALL_PIXEL
					&& lastGoodCount / (lastGoodCount + lastBadCount) > Constants.MIN_GOODPERGOODBAD_PIXEL;

			return referenceToFrame;
	}
}
