package DepthEstimation;

import java.util.Deque;
import java.util.Random;

import Utils.Constants;
import DataStructures.Frame;
import LieAlgebra.SIM3;

/**
 * Keeps a detailed depth map (consisting of DepthMapPixelHypothesis) and does
 * stereo comparisons and regularization to update it.
 */
public class DepthMap {

	
	int width;
	int height;
	
	DepthMapPixelHypothesis[] otherDepthMap;
	DepthMapPixelHypothesis[] currentDepthMap;
	
	Frame activeKeyFrame;
	boolean activeKeyFrameIsReactivated;
	
	public DepthMap(int w, int h) {
		width = w;
		height = h;
		

		activeKeyFrame = null;
		activeKeyFrameIsReactivated = false;
		otherDepthMap = new DepthMapPixelHypothesis[width*height];
		currentDepthMap = new DepthMapPixelHypothesis[width*height];
	

//		validityIntegralBuffer = (int*)Eigen::internal::aligned_malloc(width*height*sizeof(int));
//
//		debugImageHypothesisHandling = cv::Mat(h,w, CV_8UC3);
//		debugImageHypothesisPropagation = cv::Mat(h,w, CV_8UC3);
//		debugImageStereoLines = cv::Mat(h,w, CV_8UC3);
//		debugImageDepth = cv::Mat(h,w, CV_8UC3);
	
		

		reset();
		
		
		
	}

	public void reset() {
		for (int i=0 ; i<width*height ; i++) {
			otherDepthMap[i].isValid = false;
			currentDepthMap[i].isValid = false;
		}
	}
	
	public void initializeRandomly(Frame newFrame) {
		activeKeyFrame = newFrame;
		activeKeyFrameIsReactivated = false;
		
		
		Random random = new Random(System.nanoTime());
		
		float[] maxGradients = newFrame.imageGradientMaxArrayLvl[0];
		
		for(int y=1;y<height-1;y++) {
			for(int x=1;x<width-1;x++) {
				if(maxGradients[x+y*width] > Constants.MIN_ABS_GRAD_CREATE) {
					float idepth = 0.5f + 1.0f * (random.nextInt(100001) / 100000.0f);
					currentDepthMap[x+y*width] = new DepthMapPixelHypothesis(
							idepth,
							idepth,
							Constants.VAR_RANDOM_INIT_INITIAL,
							Constants.VAR_RANDOM_INIT_INITIAL,
							20);
				} else {
					currentDepthMap[x+y*width].isValid = false;
					currentDepthMap[x+y*width].blacklisted = 0;
				}
			}
		}
		

		activeKeyFrame.setDepth(currentDepthMap);
		
	}
	
	/**
	 * Return null if failed, return float[] {epx, epy} if found.
	 * */
	public float[] makeAndCheckEPL(int x, int y, Frame ref) {
		int idx = x + y*width;
		
		double fx = Constants.fx[0];
		double fy = Constants.fy[0];
		double cx = Constants.cx[0];
		double cy = Constants.cy[0];
		

		// ======= make epl ========
		// calculate the plane spanned by the two camera centers and the point (x,y,1)
		// intersect it with the keyframe's image plane (at depth=1)
		double epx = - fx * ref.thisToOther_t.get(0,0) + ref.thisToOther_t.get(2,0)*(x - cx);
		double epy = - fy * ref.thisToOther_t.get(1,0) + ref.thisToOther_t.get(2,0)*(y - cy);

		if(Double.isNaN(epx+epy)) {
			return null;
		}
		
		// ======== check epl length =========
		float eplLengthSquared = (float) (epx*epx+epy*epy);
		if(eplLengthSquared < Constants.MIN_EPL_LENGTH_SQUARED) {
			return null;
		}


		// ===== check epl-grad magnitude ======
//		float gx = activeKeyFrameImageData[idx+1] - activeKeyFrameImageData[idx-1];
//		float gy = activeKeyFrameImageData[idx+width] - activeKeyFrameImageData[idx-width];
//		float eplGradSquared = gx * epx + gy * epy;
		
		float gx = activeKeyFrame.imageGradientXArrayLvl[0][idx];
		float gy = activeKeyFrame.imageGradientYArrayLvl[0][idx];
		float eplGradSquared = activeKeyFrame.imageGradientMaxArrayLvl[0][idx];
		eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;	// square and norm with epl-length

		if(eplGradSquared < Constants.MIN_EPL_GRAD_SQUARED) {
			return null;
		}


		// ===== check epl-grad angle ======
		if(eplGradSquared / (gx*gx+gy*gy) < Constants.MIN_EPL_ANGLE_SQUARED) {
			return null;
		}


		// ===== DONE - return "normalized" epl =====
		float fac = (float) (Constants.GRADIENT_SAMPLE_DIST / Math.sqrt(eplLengthSquared));
		
		float pepx = (float) (epx * fac);
		float pepy = (float) (epy * fac);

		return new float[] {pepx, pepy};
	}
	
/*
	boolean observeDepthCreate(int x, int y, int idx) {
		DepthMapPixelHypothesis target = currentDepthMap[idx];
	
		Frame refFrame = activeKeyFrameIsReactivated ? newest_referenceFrame : oldest_referenceFrame;
	
		if(refFrame->getTrackingParent() == activeKeyFrame)
		{
			bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
			if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
			{
				return false;
			}
		}
	
	
		// Get epipolar line??
		float epx, epy;
		// x, y pixel coordinate, refFrame
		float[] epl = makeAndCheckEPL(x, y, refFrame);
		boolean isGood = (epl != null);
	
	
		if(!isGood) return false;
	
	
		float new_u = x;
		float new_v = y;
		float result_idepth, result_var, result_eplLength;
		// Do line stereo, get error, ^ results
		float error = doLineStereo(
				new_u,new_v,epx,epy,
				0.0f, 1.0f, 1.0f/MIN_DEPTH,
				refFrame, refFrame->image(0),
				result_idepth, result_var, result_eplLength, stats);
	
		if(error == -3 || error == -2)
		{
			target->blacklisted--;
		}
	
		if(error < 0 || result_var > MAX_VAR)
			return false;
		
		result_idepth = UNZERO(result_idepth);
	
		// add hypothesis
		// Set/change the hypothesis
		*target = DepthMapPixelHypothesis(
				result_idepth,
				result_var,
				VALIDITY_COUNTER_INITIAL_OBSERVE);
	
		
		return true;
	}
	*/
	/*
	public void updateKeyframe(Deque<Frame> referenceFrames) {
		//assert(isValid());

		oldest_referenceFrame = referenceFrames.front().get();
		newest_referenceFrame = referenceFrames.back().get();
		referenceFrameByID.clear();
		referenceFrameByID_offset = oldest_referenceFrame->id();
	
		// For each frame
		for(Frame frame : referenceFrames) {
			assert(frame.hasTrackingParent());

			if(frame.getTrackingParent() != activeKeyFrame) {
//				printf("WARNING: updating frame %d with %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
//						activeKeyFrame->id(), frame->id(),
//						frame->getTrackingParent()->id());
			}

			SIM3 refToKf;
			// Get SIM3
			if(frame.pose.trackingParent.frameID == activeKeyFrame.id())
				refToKf = frame.pose.thisToParent_raw;
			else
				refToKf = activeKeyFrame.getScaledCamToWorld().inverse() *  frame->getScaledCamToWorld();

			// Prepare for stereo ??
			// prepare frame for stereo with keyframe, SE3, K, level
			frame.prepareForStereoWith(activeKeyFrame, refToKf, K, 0);

			// ?? push what?
			while((int)referenceFrameByID.size() + referenceFrameByID_offset <= frame->id())
				referenceFrameByID.push_back(frame.get());
		}

		resetCounters();

		//*** OBSERVE DEPTH HERE
		observeDepth();
		//***


		// Regularize, fill holes?
		regularizeDepthMapFillHoles();

		// Regularize?
		regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

		
		// Update depth in keyframe
		if(!activeKeyFrame->depthHasBeenUpdatedFlag)
		{
			// Update keyframe with updated depth?
			activeKeyFrame->setDepth(currentDepthMap);
		}


		activeKeyFrame->numMappedOnThis++;
		activeKeyFrame->numMappedOnThisTotal++;

	}*/
}
