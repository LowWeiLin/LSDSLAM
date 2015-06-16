import org.opencv.core.Mat;

import DataStructures.Frame;
import DataStructures.FramePoseStruct;
import DataStructures.KeyFrameGraph;
import DataStructures.TrackingReference;
import DepthEstimation.DepthMap;
import LieAlgebra.SE3;


public class LSDSLAM {
	
	
	Frame currentKeyFrame;
	DepthMap map;
	Tracker tracker;
	
	boolean createNewKeyFrame = false;
	
	TrackingReference trackingReference;
	TrackingReference mappingTrackingReference;
	KeyFrameGraph keyFrameGraph;
	
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
		
		// TODO: KeyFrame graph
		//keyFrameGraph.addFrame(currentKeyFrame);

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
		
		// TODO: get initial estimate
//		SE3 frameToReference_initialEstimate = 
//				se3FromSim3(trackingReferencePose.getCamToWorld().inverse() * 
//						keyFrameGraph.allFramePoses.back().getCamToWorld());

		// Just use 0 for now.
		SE3 frameToReference_initialEstimate = SE3.exp(new double[]{0,0,0,0,0,0});

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



		keyFrameGraph->addFrame(trackingNewFrame.get());
		*/


		/*
		// Keyframe selection
		latestTrackedFrame = trackingNewFrame;
		if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
		{
			Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;
			float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT,1.0f);

			if(keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)	minVal *= 0.7;

			lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);

			if (lastTrackingClosenessScore > minVal)
			{
				createNewKeyFrame = true;

				if(enablePrintDebugInfo && printKeyframeSelectionInfo)
					printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
			}
			else
			{
				if(enablePrintDebugInfo && printKeyframeSelectionInfo)
					printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));

			}
		}


		unmappedTrackedFramesMutex.lock();
		if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
			unmappedTrackedFrames.push_back(trackingNewFrame);
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();

		// implement blocking
		if(blockUntilMapped && trackingIsGood)
		{
			boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
			while(unmappedTrackedFrames.size() > 0)
			{
				//printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
				newFrameMappedSignal.wait(lock);
			}
			lock.unlock();
		}
		*/
		
		
	}
	
	public void doMappingIteration() {
		
	}
	
}
