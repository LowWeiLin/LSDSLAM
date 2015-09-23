import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;

import jeigen.DenseMatrix;

import org.opencv.core.Mat;

import DataStructures.Frame;
import DataStructures.FramePoseStruct;
import DataStructures.KeyFrameGraph;
import DataStructures.TrackingReference;
import DepthEstimation.DepthMap;
import GlobalMapping.TrackableKeyFrameSearch;
import LieAlgebra.SE3;
import LieAlgebra.Vec;
import Utils.Constants;


public class LSDSLAM {
	
	
	Frame currentKeyFrame;
	DepthMap map;
	Tracker tracker;
	
	boolean createNewKeyFrame = false;
	
	TrackingReference trackingReference;
	TrackingReference mappingTrackingReference;
	KeyFrameGraph keyFrameGraph;
	
	Frame latestTrackedFrame;
	float lastTrackingClosenessScore;
	TrackableKeyFrameSearch trackableKeyFrameSearch = new TrackableKeyFrameSearch();
	

	// PUSHED in tracking, READ & CLEARED in mapping
	Deque<Frame> unmappedTrackedFrames = new LinkedList<Frame>();
	
	
	public LSDSLAM() {

		trackingReference = new TrackingReference();
		mappingTrackingReference = new TrackingReference();
		keyFrameGraph = new KeyFrameGraph();
		
	}
	
	public void randomInit(Mat image, int id) {
		map = new DepthMap(image.width(), image.height());
		tracker = new Tracker();
		
		// New currentKeyframe
		currentKeyFrame = new Frame(image);
		
		// Initialize map
		map.initializeRandomly(currentKeyFrame);
		
		keyFrameGraph.addFrame(currentKeyFrame);

		//TODO:
//		if(doSlam) {
//			keyFrameGraph.idToKeyFrame.insert(currentKeyFrame.id(), currentKeyFrame);
//		}
		
		
		System.out.println("Done random initialization.");
	}

	public void trackFrame(Mat image, int frameID) {
		
		Frame trackingNewFrame = new Frame(image);
		
		// TODO: implement
		/*
		if(!trackingIsGood)
		{
			relocalizer.updateCurrentFrame(trackingNewFrame);
		}*/
		
		boolean my_createNewKeyframe = createNewKeyFrame; // pre-save here, to make decision afterwards.
		if(trackingReference.keyframe != currentKeyFrame ||
				currentKeyFrame.depthHasBeenUpdatedFlag) {
			// Set tracking reference to be the currentKeyFrame
			trackingReference.importFrame(currentKeyFrame);
			currentKeyFrame.depthHasBeenUpdatedFlag = false;
		}

		FramePoseStruct trackingReferencePose = trackingReference.keyframe.pose;

		// DO TRACKING & Show tracking result.
		
		System.out.println("InitEst1: " + Arrays.toString(SE3.ln(trackingReferencePose.getCamToWorld().getSE3())));
		System.out.println("InitEst2: " + Arrays.toString(SE3.ln(keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1)
														.getCamToWorld().getSE3())));
		
		SE3 frameToReference_initialEstimate = 
				trackingReferencePose.getCamToWorld().inverse().mul(
					keyFrameGraph.allFramePoses.get(keyFrameGraph.allFramePoses.size()-1).getCamToWorld()).getSE3();

		SE3 newRefToFrame_poseUpdate = tracker.trackFrame(
				trackingReference,
				trackingNewFrame,
				frameToReference_initialEstimate);

		/*
		nTrackFrame++;

		tracking_lastResidual = tracker.lastResidual;
		tracking_lastUsage = tracker.pointUsage;
		tracking_lastGoodPerBad = tracker.lastGoodCount / (tracker.lastGoodCount + tracker.lastBadCount);
		tracking_lastGoodPerTotal = tracker.lastGoodCount / (trackingNewFrame.width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));


		if(manualTrackingLossIndicated || tracker.diverged || (keyFrameGraph.keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
		{
			printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
					trackingNewFrame->id(),
					100*tracking_lastGoodPerTotal,
					100*tracking_lastGoodPerBad,
					tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

			trackingReference->invalidate();

			trackingIsGood = false;
			nextRelocIdx = -1;

			unmappedTrackedFramesMutex.lock();
			unmappedTrackedFramesSignal.notify_one();
			unmappedTrackedFramesMutex.unlock();

			manualTrackingLossIndicated = false;
			return;
		}
		 
		 */

		keyFrameGraph.addFrame(trackingNewFrame);
		


		
		// Keyframe selection
		
		latestTrackedFrame = trackingNewFrame;
		if (!my_createNewKeyframe &&
				currentKeyFrame.numMappedOnThisTotal > Constants.MIN_NUM_MAPPED) {
			
			DenseMatrix dist = newRefToFrame_poseUpdate.getTranslationMat().mul(currentKeyFrame.meanIdepth);
			double[] distVec = Vec.vec3ToArray(dist); 
			float minVal = Math.min(0.2f + keyFrameGraph.keyframesAll.size() * 0.8f / Constants.INITIALIZATION_PHASE_COUNT, 1.0f);

			if(keyFrameGraph.keyframesAll.size() < Constants.INITIALIZATION_PHASE_COUNT) {
				minVal *= 0.7;
			}

			lastTrackingClosenessScore = trackableKeyFrameSearch.getRefFrameScore(
					(float) Vec.dot(distVec, distVec), tracker.pointUsage);

			if (lastTrackingClosenessScore > minVal) {
				//createNewKeyFrame = true;
				
				System.err.println("CREATE NEW KEYFRAME");

				//if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				//	printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
			} else {
				//if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				//	printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
			}
		}
		
		

		// Push into deque for mapping
		if(unmappedTrackedFrames.size() < 50 || 
				(unmappedTrackedFrames.size() < 100 &&
				trackingNewFrame.getTrackingParent().numMappedOnThisTotal < 10)) {
			
			unmappedTrackedFrames.push(trackingNewFrame);
		}
		
		// TODO: block till mapping is done - just do sequentially for now.
		
		
	}
	
	public boolean doMappingIteration() {
		System.out.println("Mapping!");
		if(currentKeyFrame == null) {
			System.err.println("doMappingIteration: currentKeyFrame is null!");
			return false;
		}

		//TODO:
		boolean trackingIsGood = tracker.trackingWasGood;
		
		// set mappingFrame
		if(trackingIsGood) {
			System.out.println("Tracking good!");
			if (createNewKeyFrame) {
				System.out.println("doMappingIteration: create new keyframe");
				// create new key frame
				//finishCurrentKeyframe();
				//changeKeyframe(false, true, 1.0f);
			} else {
				System.out.println("doMappingIteration: update keyframe");
				// ***Update key frame here
				boolean didSomething = updateKeyframe();

				if(!didSomething) {
					System.err.println("updateKeyFrame false");
					return false;
				}
			}
			
			// Debug and draw depth map
			//map.debugPlotDepthMap();
			
			return true;
		} else { // Tracking is not good
			System.err.println("Tracking bad!");
			/*
			
			// invalidate map if it was valid.
			if(map.isValid()) {
				if(currentKeyFrame.numMappedOnThisTotal >= MIN_NUM_MAPPED) {
					//finishCurrentKeyframe();
				} else {
					//discardCurrentKeyframe();
				}
				map.invalidate();
			}

			// start relocalizer if it isnt running already
			if(!relocalizer.isRunning)
				relocalizer.start(keyFrameGraph->keyframesAll);

			// did we find a frame to relocalize with?
			if(relocalizer.waitResult(50))
				takeRelocalizeResult();

			*/
			return true;
		}
	}
	

	// Updates key frame with measurements from a new frame.
	public boolean updateKeyframe() {
		Deque<Frame> references = new LinkedList<Frame>();

		// remove frames that have a different tracking parent.
		while(unmappedTrackedFrames.size() > 0 &&
				(!unmappedTrackedFrames.peekFirst().hasTrackingParent() ||
				unmappedTrackedFrames.peekFirst().getTrackingParent() != currentKeyFrame)) {
			System.out.println("Pop " + !unmappedTrackedFrames.peekFirst().hasTrackingParent()
					+ " " + (unmappedTrackedFrames.peekFirst().getTrackingParent() != currentKeyFrame));
			unmappedTrackedFrames.pop();//.clear_refPixelWasGood();
		}

		// clone list
		if(unmappedTrackedFrames.size() > 0) {
			// Copy from unmappedTrackedFrames to references
			references.addAll(unmappedTrackedFrames);
			
			Frame popped = unmappedTrackedFrames.pop();

			// ***DO the update here
			// references - list of frames to map
			map.updateKeyframe(references);

			//popped->clear_refPixelWasGood();
			references.clear();
		} else {
			return false;
		}



		//if(outputWrapper != 0 && continuousPCOutput && currentKeyFrame != 0)
		//	outputWrapper->publishKeyframe(currentKeyFrame.get());

		return true;
	}
	
}
