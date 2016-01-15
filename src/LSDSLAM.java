import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
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
import Tracking.Tracker;
import Utils.Constants;


public class LSDSLAM {
	
	
	Frame currentKeyFrame;
	DepthMap map;
	Tracker tracker;
	
	boolean createNewKeyFrame = false;
	boolean trackingIsGood = true;
	
	final boolean SLAMEnabled = true;
	
	TrackingReference trackingReference;
	TrackingReference mappingTrackingReference;
	KeyFrameGraph keyFrameGraph;
	
	Frame latestTrackedFrame;
	float lastTrackingClosenessScore;
	TrackableKeyFrameSearch trackableKeyFrameSearch = null;

	// PUSHED in tracking, READ & CLEARED in mapping
	Deque<Frame> unmappedTrackedFrames = new LinkedList<Frame>();
	
	// PUSHED by Mapping, READ & CLEARED by constraintFinder
	Deque<Frame> newKeyFrames = new LinkedList<Frame>();
	
	int nextRelocIdx = -1;
	
	public LSDSLAM() {

		trackingReference = new TrackingReference();
		mappingTrackingReference = new TrackingReference();
		keyFrameGraph = new KeyFrameGraph();
		
	}
	
	public void randomInit(Mat image, int id) {
		
		if (trackableKeyFrameSearch == null) {
			trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph, image.width(), image.height());
		}
		
		map = new DepthMap(image.width(), image.height());
		tracker = new Tracker();
		
		// New currentKeyframe
		currentKeyFrame = new Frame(image);
		currentKeyFrame.isKF = true;
		
		// Initialize map
		map.initializeRandomly(currentKeyFrame);
		
		keyFrameGraph.addFrame(currentKeyFrame);

		if(SLAMEnabled) {
			keyFrameGraph.idToKeyFrame.put(currentKeyFrame.id(), currentKeyFrame);
		}
		
		System.out.println("Done random initialization.");
	}

	public void trackFrame(Mat image, int frameID) {
		
		Frame trackingNewFrame = new Frame(image);
		
		// TODO: implement
		if(!trackingIsGood)
		{
			System.err.println("RELOCALIZE");
			//relocalizer.updateCurrentFrame(trackingNewFrame);
		}
		
		boolean my_createNewKeyframe = createNewKeyFrame; // pre-save here, to make decision afterwards.
		if(trackingReference.keyframe != currentKeyFrame ||
				currentKeyFrame.depthHasBeenUpdatedFlag) {
			// Set tracking reference to be the currentKeyFrame
			trackingReference.importFrame(currentKeyFrame);
			currentKeyFrame.depthHasBeenUpdatedFlag = false;
		}

		FramePoseStruct trackingReferencePose = trackingReference.keyframe.pose;

		// DO TRACKING & Show tracking result.
		
		/*
		System.out.println("trackingReferencePose.getCamToWorld() : " 
		+ trackingReferencePose.getCamToWorld().toString());
		System.out.println(keyFrameGraph.allFramePoses
				.get(keyFrameGraph.allFramePoses.size()-1).getCamToWorld());
		*/
		
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


		 */
		
		if(Constants.manualTrackingLossIndicated || tracker.diverged || (keyFrameGraph.keyframesAll.size() > Constants.INITIALIZATION_PHASE_COUNT && !tracker.trackingWasGood))
		{
			System.out.println("manualTrackingLossIndicated: " + Boolean.toString(Constants.manualTrackingLossIndicated));
			System.out.println("keyFrameGraph.keyframesAll.size(): " + keyFrameGraph.keyframesAll.size());
			System.out.println("tracker.trackingWasGood: " + Boolean.toString(tracker.trackingWasGood));
			System.out.println("tracker.diverged: " + Boolean.toString(tracker.diverged));
			
			/*
			printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
					trackingNewFrame.id(),
					100*tracking_lastGoodPerTotal,
					100*tracking_lastGoodPerBad,
					tracker.diverged ? "DIVERGED" : "NOT DIVERGED");
			*/
			trackingReference.invalidate();

			trackingIsGood = false;
			nextRelocIdx = -1;

			Constants.manualTrackingLossIndicated = false;
			return;
		}
		 

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
				System.err.println("CREATE NEW KEYFRAME");
				createNewKeyFrame = true;
				trackingNewFrame.isKF = true;
			} else {
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
		if(currentKeyFrame == null) {
			System.err.println("doMappingIteration: currentKeyFrame is null!");
			return false;
		}

		boolean trackingIsGood = tracker.trackingWasGood;
		
		// set mappingFrame
		if(trackingIsGood) {
			System.out.println("Tracking was good!");
			if (createNewKeyFrame) {
				System.out.println("doMappingIteration: create new keyframe");
				// create new key frame
				finishCurrentKeyframe();
				changeKeyframe(false, true, 1.0f);
			} else {
				System.out.println("doMappingIteration: update keyframe");
				// ***Update key frame here***
				boolean didSomething = updateKeyframe();

				if(!didSomething) {
					System.err.println("updateKeyFrame: false");
					return false;
				}
			}
			
			return true;
		} else { // Tracking is not good
			System.err.println("Tracking was bad!");
			
			
			// TODO: relocalize
			System.err.println("Relocalize (2)");
			
			
			// invalidate map if it was valid.
			if(map.isValid()) {
				if(currentKeyFrame.numMappedOnThisTotal >= Constants.MIN_NUM_MAPPED) {
					finishCurrentKeyframe();
				} else {
					System.err.println("DiscardCurrentkeyFrame");
					//discardCurrentKeyframe();
				}
				System.err.println("map.invalidate");
				//map.invalidate();
			}
			/*
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

			// ***Do the update here***
			// references - list of frames to map
			map.updateKeyframe(references);

			popped.clear_refPixelWasGood();
			
			// CLEAR DATA
			for (Frame f : references) {
				if (!createNewKeyFrame && f.isKF == false) {
					f.clearData();
				}
			}
			
			references.clear();
			unmappedTrackedFrames.clear();
			
		} else {
			return false;
		}

		return true;
	}

	void finishCurrentKeyframe()
	{
		System.out.println("FINALIZING KF: " + currentKeyFrame.id());
	
		map.finalizeKeyFrame();
	
	
		if(SLAMEnabled) {
			mappingTrackingReference.importFrame(currentKeyFrame);
			currentKeyFrame.setPermaRef(mappingTrackingReference);
			mappingTrackingReference.invalidate();
	
			if(currentKeyFrame.idxInKeyframes < 0) {
				currentKeyFrame.idxInKeyframes = keyFrameGraph.keyframesAll.size();
				keyFrameGraph.keyframesAll.add(currentKeyFrame);
				keyFrameGraph.totalPoints += currentKeyFrame.numPoints;
				keyFrameGraph.totalVertices++;
	
				newKeyFrames.add(currentKeyFrame);
			}
		}
		
		// WRITE POINT CLOUD TO FILE
		try {
			keyFrameGraph.writePointCloudToFile("graphPOINTCLOUD-" + currentKeyFrame.id() + ".xyz");
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
		}
		
		//if(outputWrapper!= 0)
		//	outputWrapper->publishKeyframe(currentKeyFrame.get());
	}
	

	void changeKeyframe(boolean noCreate, boolean force, float maxScore) {
		Frame newReferenceKF = null;
		Frame newKeyframeCandidate = latestTrackedFrame;
		
		if(Constants.doKFReActivation && SLAMEnabled)
		{
			// TODO: use relocalization
			//newReferenceKF = trackableKeyFrameSearch.findRePositionCandidate(newKeyframeCandidate, maxScore);
		}
	
		if(newReferenceKF != null) {
			System.err.println("RELOCALIZED!");
			loadNewCurrentKeyframe(newReferenceKF);
		} else {
			if(force) {
				if(noCreate) {
					trackingIsGood = false;
					nextRelocIdx = -1;
					System.out.println("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
				} else {
					createNewCurrentKeyframe(newKeyframeCandidate);
				}
			}
		}
	
	
		createNewKeyFrame = false;
	}
	
	void loadNewCurrentKeyframe(Frame keyframeToLoad)
	{
		System.out.printf("RE-ACTIVATE KF %d\n", keyframeToLoad.id());

		map.setFromExistingKF(keyframeToLoad);

		System.out.printf("re-activate frame %d!\n", keyframeToLoad.id());

		currentKeyFrame = keyFrameGraph.idToKeyFrame.get(keyframeToLoad.id());
		currentKeyFrame.depthHasBeenUpdatedFlag = false;
	}
	
	void createNewCurrentKeyframe(Frame newKeyframeCandidate)
	{
		System.out.printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate.id(), currentKeyFrame.id());


		if(SLAMEnabled)
		{
			// add NEW keyframe to id-lookup
			keyFrameGraph.idToKeyFrame.put(newKeyframeCandidate.id(), newKeyframeCandidate);
		}

		// propagate & make new.
		map.createKeyFrame(newKeyframeCandidate);

		/*
		if(printPropagationStatistics)
		{

			Eigen::Matrix<float, 20, 1> data;
			data.setZero();
			data[0] = runningStats.num_prop_attempts / ((float)width*height);
			data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
			data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

			outputWrapper->publishDebugInfo(data);
		}
		*/

		currentKeyFrame = newKeyframeCandidate;
	}
}
